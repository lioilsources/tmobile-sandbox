package main

// Tenanti: API klíč → tenant, limit souběhu, měření tokenů (i ze streamu), kniha pro účtování (JSONL)
// a metriky pro Prometheus. Bez externích závislostí — stejně jako zbytek gateway.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Price je ceník modelu v Kč: za 1M vstupních / výstupních tokenů, nebo za minutu audia.
type Price struct {
	InPerM  float64 `json:"in_per_m"`
	OutPerM float64 `json:"out_per_m"`
	PerMin  float64 `json:"per_audio_min"`
}

type Tenant struct {
	Name           string   `json:"name"`
	Keys           []string `json:"keys"`
	MaxConcurrency int      `json:"max_concurrency"`
	Models         []string `json:"models"` // prázdné = všechny
	sem            chan struct{}
}

type TenantConfig struct {
	Tenants []*Tenant        `json:"tenants"`
	Prices  map[string]Price `json:"prices"`
}

type tenantKey struct{}

// Usage je jeden řádek knihy pro účtování.
type Usage struct {
	TS               time.Time `json:"ts"`
	Tenant           string    `json:"tenant"`
	Model            string    `json:"model"`
	Path             string    `json:"path"`
	Status           int       `json:"status"`
	Stream           bool      `json:"stream"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	AudioSeconds     float64   `json:"audio_s,omitempty"`
	CostCZK          float64   `json:"cost_czk"`
	MS               int64     `json:"ms"`
	TTFTMS           int64     `json:"ttft_ms,omitempty"`
	ReqID            any       `json:"req_id"`
}

type tenantStore struct {
	cfg         TenantConfig
	upstreamKey string // klíč, kterým se gateway prokazuje upstreamu (LiteLLM master key); klíč tenanta dál nejde
	byKey       map[string]*Tenant
	ledger      *os.File
	mu          sync.Mutex
	m           *metrics
}

func loadTenants(path, ledgerPath string) (*tenantStore, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg TenantConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	s := &tenantStore{cfg: cfg, byKey: map[string]*Tenant{}, m: newMetrics(), upstreamKey: os.Getenv("UPSTREAM_API_KEY")}
	for _, t := range cfg.Tenants {
		if t.MaxConcurrency <= 0 {
			t.MaxConcurrency = 64
		}
		t.sem = make(chan struct{}, t.MaxConcurrency)
		for _, k := range t.Keys {
			s.byKey[k] = t
		}
	}
	if s.ledger, err = os.OpenFile(ledgerPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *tenantStore) cost(model string, u *Usage) float64 {
	p, ok := s.cfg.Prices[model]
	if !ok {
		return 0
	}
	return float64(u.PromptTokens)/1e6*p.InPerM + float64(u.CompletionTokens)/1e6*p.OutPerM + u.AudioSeconds/60*p.PerMin
}

func (s *tenantStore) record(u *Usage) {
	u.CostCZK = s.cost(u.Model, u)
	s.m.observe(u)
	line, _ := json.Marshal(u)
	s.mu.Lock()
	s.ledger.Write(append(line, '\n')) //nolint:errcheck
	s.mu.Unlock()
}

func apiError(w http.ResponseWriter, code int, typ, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"message": msg, "type": typ}}) //nolint:errcheck
}

// withTenant ověří klíč, hlídá limit souběhu, doplní stream_options.include_usage a změří odpověď.
func (s *tenantStore) withTenant(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		t := s.byKey[key]
		if t == nil {
			s.m.reject("unknown", "auth")
			apiError(w, http.StatusUnauthorized, "invalid_api_key", "neplatný API klíč")
			return
		}
		select {
		case t.sem <- struct{}{}:
			defer func() { <-t.sem }()
		default:
			s.m.reject(t.Name, "concurrency")
			w.Header().Set("Retry-After", "1")
			apiError(w, http.StatusTooManyRequests, "rate_limit", fmt.Sprintf("tenant %s: limit %d souběžných požadavků", t.Name, t.MaxConcurrency))
			return
		}
		s.m.inflight(t.Name, +1)
		defer s.m.inflight(t.Name, -1)

		u := &Usage{TS: time.Now(), Tenant: t.Name, Path: r.URL.Path, ReqID: r.Context().Value(reqIDKey)}
		if strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") && r.Body != nil {
			body, _ := io.ReadAll(r.Body)
			var req map[string]any
			if json.Unmarshal(body, &req) == nil {
				u.Model, _ = req["model"].(string)
				u.Stream, _ = req["stream"].(bool)
				if u.Stream && strings.HasSuffix(r.URL.Path, "/completions") {
					req["stream_options"] = map[string]any{"include_usage": true}
					body, _ = json.Marshal(req)
				}
			}
			r.Body = io.NopCloser(bytes.NewReader(body))
			r.ContentLength = int64(len(body))
			r.Header.Set("Content-Length", strconv.Itoa(len(body)))
		} else if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
			if err := r.ParseMultipartForm(64 << 20); err == nil {
				u.Model = r.FormValue("model")
				r = rebuildMultipart(r) // tělo už bylo přečteno — sestavit znovu pro upstream
			}
		}
		if len(t.Models) > 0 && u.Model != "" && !contains(t.Models, u.Model) {
			s.m.reject(t.Name, "model")
			apiError(w, http.StatusForbidden, "model_not_allowed", fmt.Sprintf("tenant %s nemá přístup k modelu %s", t.Name, u.Model))
			return
		}
		r.Header.Del("Authorization") // klíč tenanta upstream neuvidí
		if s.upstreamKey != "" {
			r.Header.Set("Authorization", "Bearer "+s.upstreamKey)
		}
		r.Header.Set("X-Tenant", t.Name)

		mw := &meterWriter{ResponseWriter: w, start: u.TS, stream: u.Stream}
		next.ServeHTTP(mw, r)
		u.Status = mw.status()
		u.MS = time.Since(u.TS).Milliseconds()
		if !mw.firstByte.IsZero() && u.Stream {
			u.TTFTMS = mw.firstByte.Sub(u.TS).Milliseconds()
		}
		mw.parseUsage(u)
		s.record(u)
	})
}

func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

// rebuildMultipart znovu sestaví multipart tělo z už rozparsovaného formuláře (model jsme museli přečíst).
func rebuildMultipart(r *http.Request) *http.Request {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	for k, vs := range r.MultipartForm.Value {
		for _, v := range vs {
			mw.WriteField(k, v) //nolint:errcheck
		}
	}
	for k, fhs := range r.MultipartForm.File {
		for _, fh := range fhs {
			f, err := fh.Open()
			if err != nil {
				continue
			}
			part, _ := mw.CreateFormFile(k, fh.Filename)
			io.Copy(part, f) //nolint:errcheck
			f.Close()
		}
	}
	mw.Close()
	r.Body = io.NopCloser(&buf)
	r.ContentLength = int64(buf.Len())
	r.Header.Set("Content-Length", strconv.Itoa(buf.Len()))
	r.Header.Set("Content-Type", mw.FormDataContentType())
	return r
}

// meterWriter propouští odpověď beze změny a zároveň si drží, co potřebuje k měření:
// celé JSON tělo (do 16 MB), u SSE jen poslední řádky „data:“ s usage.
type meterWriter struct {
	http.ResponseWriter
	code      int
	start     time.Time
	firstByte time.Time
	stream    bool
	body      bytes.Buffer
	lastUsage []byte
	partial   []byte
}

func (m *meterWriter) WriteHeader(code int)        { m.code = code; m.ResponseWriter.WriteHeader(code) }
func (m *meterWriter) Unwrap() http.ResponseWriter { return m.ResponseWriter }

// Flush musí projít celým řetězcem obalů (statusWriter z withLogging Flush nemá) — jinak se SSE bufferuje po 4 KB.
func (m *meterWriter) Flush() { http.NewResponseController(m.ResponseWriter).Flush() } //nolint:errcheck

func (m *meterWriter) status() int {
	if m.code == 0 {
		return http.StatusOK
	}
	return m.code
}

func (m *meterWriter) Write(p []byte) (int, error) {
	if m.firstByte.IsZero() {
		m.firstByte = time.Now()
	}
	if m.stream {
		data := append(m.partial, p...)
		sc := bufio.NewScanner(bytes.NewReader(data))
		sc.Buffer(make([]byte, 0, 64<<10), 1<<20)
		var lines [][]byte
		for sc.Scan() {
			lines = append(lines, append([]byte(nil), sc.Bytes()...))
		}
		m.partial = nil
		if len(data) > 0 && data[len(data)-1] != '\n' && len(lines) > 0 { // poslední řádek ještě není celý
			m.partial = lines[len(lines)-1]
			lines = lines[:len(lines)-1]
		}
		for _, l := range lines {
			if bytes.HasPrefix(l, []byte("data: ")) && bytes.Contains(l, []byte(`"usage":{`)) {
				m.lastUsage = l[6:]
			}
		}
	} else if m.body.Len() < 16<<20 {
		m.body.Write(p)
	}
	return m.ResponseWriter.Write(p)
}

func (m *meterWriter) parseUsage(u *Usage) {
	src := m.body.Bytes()
	if m.stream {
		src = m.lastUsage
	}
	var resp struct {
		Model string `json:"model"`
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
		Duration float64 `json:"duration"`
	}
	if json.Unmarshal(src, &resp) != nil {
		return
	}
	if resp.Usage != nil {
		u.PromptTokens, u.CompletionTokens = resp.Usage.PromptTokens, resp.Usage.CompletionTokens
	}
	u.AudioSeconds = resp.Duration
	if u.Model == "" {
		u.Model = resp.Model
	}
}

// --- metriky (Prometheus text format, ručně — gateway nemá závislosti) ---

var buckets = []float64{0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120}

type hist struct {
	counts []uint64
	sum    float64
	n      uint64
}

type metrics struct {
	mu       sync.Mutex
	counters map[string]float64 // klíč = "name{labels}"
	gauges   map[string]float64
	hists    map[string]*hist
}

func newMetrics() *metrics {
	return &metrics{counters: map[string]float64{}, gauges: map[string]float64{}, hists: map[string]*hist{}}
}

func lbl(kv ...string) string {
	var b strings.Builder
	for i := 0; i < len(kv); i += 2 {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%s=%q", kv[i], kv[i+1])
	}
	return b.String()
}

func (m *metrics) add(name, labels string, v float64) {
	m.counters[name+"{"+labels+"}"] += v
}

func (m *metrics) observeHist(name, labels string, v float64) {
	k := name + "{" + labels + "}"
	h := m.hists[k]
	if h == nil {
		h = &hist{counts: make([]uint64, len(buckets))}
		m.hists[k] = h
	}
	for i, b := range buckets {
		if v <= b {
			h.counts[i]++
		}
	}
	h.sum += v
	h.n++
}

func (m *metrics) reject(tenant, reason string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.add("aigw_rejected_total", lbl("tenant", tenant, "reason", reason), 1)
}

func (m *metrics) inflight(tenant string, d float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.gauges["aigw_inflight{"+lbl("tenant", tenant)+"}"] += d
}

func (m *metrics) observe(u *Usage) {
	m.mu.Lock()
	defer m.mu.Unlock()
	tm := lbl("tenant", u.Tenant, "model", u.Model)
	m.add("aigw_requests_total", lbl("tenant", u.Tenant, "model", u.Model, "status", strconv.Itoa(u.Status)), 1)
	m.add("aigw_tokens_total", tm+`,kind="prompt"`, float64(u.PromptTokens))
	m.add("aigw_tokens_total", tm+`,kind="completion"`, float64(u.CompletionTokens))
	m.add("aigw_audio_seconds_total", tm, u.AudioSeconds)
	m.add("aigw_cost_czk_total", tm, u.CostCZK)
	m.observeHist("aigw_request_duration_seconds", tm, float64(u.MS)/1000)
	if u.TTFTMS > 0 {
		m.observeHist("aigw_ttft_seconds", tm, float64(u.TTFTMS)/1000)
	}
}

func (m *metrics) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
	m.mu.Lock()
	defer m.mu.Unlock()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	for k, v := range m.counters {
		fmt.Fprintf(w, "%s %g\n", k, v)
	}
	for k, v := range m.gauges {
		fmt.Fprintf(w, "%s %g\n", k, v)
	}
	for k, h := range m.hists {
		name, labels := k[:strings.IndexByte(k, '{')], k[strings.IndexByte(k, '{')+1:len(k)-1]
		for i, b := range buckets {
			fmt.Fprintf(w, "%s_bucket{%s,le=\"%g\"} %d\n", name, labels, b, h.counts[i])
		}
		fmt.Fprintf(w, "%s_bucket{%s,le=\"+Inf\"} %d\n%s_sum{%s} %g\n%s_count{%s} %d\n", name, labels, h.n, name, labels, h.sum, name, labels, h.n)
	}
	slog.Debug("metrics scraped")
}
