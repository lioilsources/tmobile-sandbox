// qctl: GPU job dispatcher for the L40S VM.
//
// Scheduling rule: short jobs first, long "filler" jobs fill idle time.
//   - Non-filler jobs run in ascending est_min order (ties: submit time) and are never preempted.
//   - Filler jobs run only when no non-filler job is pending; when one arrives the filler
//     gets SIGTERM (process group), must checkpoint and exit, and goes back to pending.
//
// Job kinds:
//   - cmd:    qctl runs `cmd` under bash in its own session, stdout/stderr to logs/<name>.log.
//   - window: qctl switches the GPU profile, waits for `health`, writes the READY marker and
//     waits until another session writes the DONE marker (or guard_min runs out).
//
// State lives in $QCTL_DIR (default /srv/work/queue): jobs/{pending,running,done,failed}/<name>.json.
// A HOLD file in $QCTL_DIR stops new starts (running jobs continue).
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type Job struct {
	Name     string `json:"name"`
	Kind     string `json:"kind"` // cmd | window
	Cmd      string `json:"cmd,omitempty"`
	Profile  string `json:"profile"` // switch.sh profile: A B C D L O S W off
	EstMin   int    `json:"est_min"`
	Filler   bool   `json:"filler,omitempty"`
	Ready    string `json:"ready,omitempty"`  // window: marker file name in QCTL_DIR
	Done     string `json:"done,omitempty"`   // window: marker file name in QCTL_DIR
	Health   string `json:"health,omitempty"` // window: URL polled before READY
	GuardMin int    `json:"guard_min,omitempty"`

	Submitted time.Time `json:"submitted"`
	Started   time.Time `json:"started,omitempty"`
	Finished  time.Time `json:"finished,omitempty"`
	PID       int       `json:"pid,omitempty"`
	Preempted int       `json:"preempted,omitempty"` // how many times
	Stopping  bool      `json:"stopping,omitempty"`  // SIGTERM sent by preemption
	StopAt    time.Time `json:"stop_at,omitempty"`
	Exit      *int      `json:"exit,omitempty"`
	Note      string    `json:"note,omitempty"`
}

var (
	dir      = env("QCTL_DIR", "/srv/work/queue")
	switchSh = env("QCTL_SWITCH", os.Getenv("HOME")+"/tmobile-sandbox/50_vllm/switch.sh")
	profFile = env("QCTL_PROFILE_FILE", "/srv/work/.gpu_profile")
	tick     = 10 * time.Second
	killWait = 15 * time.Minute // SIGTERM → SIGKILL for a preempted filler
)

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func stateDir(s string) string      { return filepath.Join(dir, "jobs", s) }
func jobPath(s, name string) string { return filepath.Join(stateDir(s), name+".json") }
func logPath(name string) string    { return filepath.Join(dir, "logs", name+".log") }
func exitPath(name string) string   { return filepath.Join(dir, "logs", name+".exit") }

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	for _, s := range []string{"pending", "running", "done", "failed"} {
		must(os.MkdirAll(stateDir(s), 0o755))
	}
	must(os.MkdirAll(filepath.Join(dir, "logs"), 0o755))
	switch os.Args[1] {
	case "daemon":
		daemon()
	case "add":
		if len(os.Args) != 3 {
			usage()
		}
		add(os.Args[2])
	case "ls":
		ls()
	case "done":
		if len(os.Args) != 3 {
			usage()
		}
		markDone(os.Args[2])
	case "cancel":
		if len(os.Args) != 3 {
			usage()
		}
		cancel(os.Args[2])
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage: qctl daemon | add <job.json|-> | ls | done <name> | cancel <name>`)
	os.Exit(2)
}

func must(err error) {
	if err != nil {
		fatalf("%v", err)
	}
}

func logf(format string, a ...any) { slog.Info(fmt.Sprintf(format, a...)) }

func fatalf(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "qctl: "+format+"\n", a...)
	os.Exit(1)
}

// ---------- CLI ----------

func add(src string) {
	var b []byte
	var err error
	if src == "-" {
		b, err = io.ReadAll(os.Stdin)
	} else {
		b, err = os.ReadFile(src)
	}
	must(err)
	var j Job
	must(json.Unmarshal(b, &j))
	if err := validate(&j); err != nil {
		fatalf("%v", err)
	}
	for _, s := range []string{"pending", "running"} {
		if _, err := os.Stat(jobPath(s, j.Name)); err == nil {
			fatalf("job %s already %s", j.Name, s)
		}
	}
	j.Submitted = time.Now()
	must(save("pending", &j))
	fmt.Println(j.Name)
}

func validate(j *Job) error {
	if j.Name == "" || strings.ContainsAny(j.Name, "/ ") {
		return errors.New("name required, no spaces or slashes")
	}
	if j.Profile == "" {
		j.Profile = "off"
	}
	if j.EstMin <= 0 {
		return errors.New("est_min required (minutes)")
	}
	switch j.Kind {
	case "cmd":
		if j.Cmd == "" {
			return errors.New("cmd job needs cmd")
		}
	case "window":
		if j.Ready == "" || j.Done == "" {
			return errors.New("window job needs ready and done markers")
		}
		if j.Filler {
			return errors.New("window job cannot be filler")
		}
		if j.GuardMin <= 0 {
			j.GuardMin = j.EstMin * 2
		}
	default:
		return errors.New("kind must be cmd or window")
	}
	return nil
}

func ls() {
	for _, s := range []string{"running", "pending", "done", "failed"} {
		js := load(s)
		if s == "pending" {
			sortQueue(js)
		}
		if s == "done" || s == "failed" {
			sort.Slice(js, func(a, b int) bool { return js[a].Finished.After(js[b].Finished) })
			if len(js) > 8 {
				js = js[:8]
			}
		}
		for _, j := range js {
			t := j.Submitted
			if !j.Started.IsZero() {
				t = j.Started
			}
			if !j.Finished.IsZero() {
				t = j.Finished
			}
			cls := "short"
			if j.Filler {
				cls = "filler"
			}
			ex := ""
			if j.Exit != nil {
				ex = " exit=" + strconv.Itoa(*j.Exit)
			}
			fmt.Printf("%-8s %-28s %-6s %-6s %4d min  %s%s %s\n", s, j.Name, j.Kind, cls, j.EstMin,
				t.Format("01-02 15:04"), ex, j.Note)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "HOLD")); err == nil {
		fmt.Println("HOLD: new starts paused")
	}
}

func markDone(name string) {
	j, err := read(jobPath("running", name))
	must(err)
	if j.Kind != "window" {
		fatalf("done is only for window jobs; cmd jobs finish on exit")
	}
	must(os.WriteFile(filepath.Join(dir, j.Done), nil, 0o644))
}

func cancel(name string) {
	if j, err := read(jobPath("pending", name)); err == nil {
		j.Note = "cancelled"
		j.Finished = time.Now()
		must(move("pending", "failed", j))
		return
	}
	j, err := read(jobPath("running", name))
	must(err)
	if j.Kind == "window" {
		must(os.WriteFile(filepath.Join(dir, j.Done), nil, 0o644))
		return
	}
	j.Note = "cancelled"
	must(save("running", j))
	_ = syscall.Kill(-j.PID, syscall.SIGTERM)
}

// ---------- storage ----------

func read(p string) (*Job, error) {
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var j Job
	return &j, json.Unmarshal(b, &j)
}

func load(s string) []*Job {
	ms, _ := filepath.Glob(filepath.Join(stateDir(s), "*.json"))
	var js []*Job
	for _, m := range ms {
		if j, err := read(m); err == nil {
			js = append(js, j)
		} else {
			logf("skip %s: %v", m, err)
		}
	}
	return js
}

func save(s string, j *Job) error {
	b, _ := json.MarshalIndent(j, "", "  ")
	tmp := jobPath(s, j.Name) + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, jobPath(s, j.Name))
}

func move(from, to string, j *Job) error {
	if err := save(to, j); err != nil {
		return err
	}
	return os.Remove(jobPath(from, j.Name))
}

// sortQueue: non-filler first, then shortest est_min, then oldest.
func sortQueue(js []*Job) {
	sort.SliceStable(js, func(a, b int) bool {
		x, y := js[a], js[b]
		if x.Filler != y.Filler {
			return !x.Filler
		}
		if x.EstMin != y.EstMin {
			return x.EstMin < y.EstMin
		}
		return x.Submitted.Before(y.Submitted)
	})
}

// ---------- daemon ----------

func daemon() {
	lf, err := os.OpenFile(filepath.Join(dir, "qctl.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	must(err)
	slog.SetDefault(slog.New(slog.NewTextHandler(lf, nil)))
	logf("daemon start (dir %s)", dir)
	held := false
	for {
		step(&held)
		time.Sleep(tick)
	}
}

func step(held *bool) {
	pending := load("pending")
	sortQueue(pending)
	running := load("running")

	for _, j := range running {
		switch j.Kind {
		case "cmd":
			checkCmd(j, pending)
		case "window":
			checkWindow(j)
		}
	}
	if len(load("running")) > 0 || len(pending) == 0 {
		return
	}
	if _, err := os.Stat(filepath.Join(dir, "HOLD")); err == nil {
		if !*held {
			logf("HOLD present, not starting %s", pending[0].Name)
			*held = true
		}
		return
	}
	*held = false
	start(pending[0])
}

func alive(pid int) bool { return pid > 0 && syscall.Kill(pid, 0) == nil }

func checkCmd(j *Job, pending []*Job) {
	if alive(j.PID) {
		if j.Filler && !j.Stopping && len(pending) > 0 && !pending[0].Filler {
			logf("preempt filler %s for %s (SIGTERM)", j.Name, pending[0].Name)
			j.Stopping, j.StopAt = true, time.Now()
			must(save("running", j))
			_ = syscall.Kill(-j.PID, syscall.SIGTERM)
		} else if j.Stopping && time.Since(j.StopAt) > killWait {
			logf("filler %s ignored SIGTERM for %v, SIGKILL", j.Name, killWait)
			_ = syscall.Kill(-j.PID, syscall.SIGKILL)
		}
		return
	}
	// The process group may still hold GPU children; wait for them.
	if syscall.Kill(-j.PID, 0) == nil {
		return
	}
	code := -1
	if b, err := os.ReadFile(exitPath(j.Name)); err == nil {
		code, _ = strconv.Atoi(strings.TrimSpace(string(b)))
	}
	_ = os.Remove(exitPath(j.Name))
	j.PID = 0
	if j.Stopping && j.Note != "cancelled" {
		j.Stopping = false
		j.Preempted++
		j.Started = time.Time{}
		logf("filler %s stopped (exit %d), back to pending", j.Name, code)
		must(move("running", "pending", j))
		return
	}
	j.Exit, j.Finished = &code, time.Now()
	to := "done"
	if code != 0 {
		to = "failed"
	}
	logf("%s finished exit %d → %s", j.Name, code, to)
	must(move("running", to, j))
}

func checkWindow(j *Job) {
	donePath := filepath.Join(dir, j.Done)
	_, err := os.Stat(donePath)
	marker := err == nil
	guard := time.Since(j.Started) > time.Duration(j.GuardMin)*time.Minute
	if !marker && !guard {
		return
	}
	j.Finished = time.Now()
	code := 0
	if marker {
		j.Note = "done marker"
	} else {
		j.Note = "guard expired"
		code = 1
	}
	j.Exit = &code
	_ = os.Remove(filepath.Join(dir, j.Ready))
	_ = os.Remove(donePath)
	logf("window %s closed (%s)", j.Name, j.Note)
	to := "done"
	if code != 0 {
		to = "failed"
	}
	must(move("running", to, j))
}

func currentProfile() string {
	b, _ := os.ReadFile(profFile)
	f := strings.Fields(string(b))
	if len(f) == 0 {
		return ""
	}
	return f[0]
}

func gpuBusy() bool {
	out, err := exec.Command("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader").Output()
	return err == nil && strings.TrimSpace(string(out)) != ""
}

func ensureProfile(p string) error {
	if currentProfile() == p {
		return nil
	}
	logf("switch profile %s → %s", currentProfile(), p)
	lf, err := os.OpenFile(filepath.Join(dir, "qctl.log"), os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer lf.Close()
	c := exec.Command(switchSh, p)
	c.Stdout, c.Stderr = lf, lf
	if err := c.Run(); err != nil {
		return err
	}
	if p == "off" {
		_ = os.WriteFile(profFile, []byte("off\n"), 0o644)
		for i := 0; i < 60 && gpuBusy(); i++ {
			time.Sleep(5 * time.Second)
		}
		if gpuBusy() {
			return errors.New("GPU still busy after switching off")
		}
	}
	return nil
}

func start(j *Job) {
	if err := ensureProfile(j.Profile); err != nil {
		logf("%s: profile %s failed: %v", j.Name, j.Profile, err)
		j.Note, j.Finished = "profile switch failed: "+err.Error(), time.Now()
		must(move("pending", "failed", j))
		return
	}
	j.Started, j.Note = time.Now(), ""
	switch j.Kind {
	case "cmd":
		lf, err := os.OpenFile(logPath(j.Name), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		must(err)
		fmt.Fprintf(lf, "\n=== qctl start %s ===\n", j.Started.Format(time.RFC3339))
		// The wrapper records the exit code so a restarted daemon can still read it.
		c := exec.Command("bash", "-c", j.Cmd+"\necho $? > "+shellQuote(exitPath(j.Name)))
		c.Dir = os.Getenv("HOME")
		c.Stdin, c.Stdout, c.Stderr = nil, lf, lf
		c.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
		if err := c.Start(); err != nil {
			lf.Close()
			j.Note, j.Finished = "start failed: "+err.Error(), time.Now()
			must(move("pending", "failed", j))
			return
		}
		lf.Close()
		j.PID = c.Process.Pid
		go func() { _ = c.Wait() }() // reap
		logf("start cmd %s pid %d (profile %s, est %d min, filler %v)", j.Name, j.PID, j.Profile, j.EstMin, j.Filler)
		must(move("pending", "running", j))
	case "window":
		must(move("pending", "running", j))
		_ = os.Remove(filepath.Join(dir, j.Done))
		if j.Health != "" {
			ok := false
			for i := 0; i < 90; i++ {
				if r, err := http.Get(j.Health); err == nil {
					r.Body.Close()
					if r.StatusCode < 400 {
						ok = true
						break
					}
				}
				time.Sleep(10 * time.Second)
			}
			if !ok {
				logf("window %s: health %s not ready after 15 min", j.Name, j.Health)
			}
		}
		must(os.WriteFile(filepath.Join(dir, j.Ready), nil, 0o644))
		logf("window %s READY (%s, profile %s, guard %d min)", j.Name, j.Ready, j.Profile, j.GuardMin)
	}
}

func shellQuote(s string) string { return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'" }
