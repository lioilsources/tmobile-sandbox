package main

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Fake upstream: JSON odpověď s usage, SSE stream s usage v posledním chunku (jen když klient poslal include_usage).
func fakeUpstream(t *testing.T) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Error("upstream nemá dostat klíč tenanta")
		}
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req) //nolint:errcheck
		if req["stream"] == true {
			if so, _ := req["stream_options"].(map[string]any); so["include_usage"] != true {
				t.Error("gateway nedoplnila stream_options.include_usage")
			}
			w.Header().Set("Content-Type", "text/event-stream")
			for _, c := range []string{`{"choices":[{"delta":{"content":"Ahoj"}}]}`, `{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7}}`} {
				io.WriteString(w, "data: "+c+"\n\n") //nolint:errcheck
				w.(http.Flusher).Flush()
			}
			io.WriteString(w, "data: [DONE]\n\n") //nolint:errcheck
			return
		}
		io.WriteString(w, `{"model":"chat","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":50}}`) //nolint:errcheck
	}))
}

func TestTenantMetering(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"tenants":[{"name":"A","keys":["ka"],"max_concurrency":2},{"name":"B","keys":["kb"],"models":["embed"]}],
	         "prices":{"chat":{"in_per_m":10,"out_per_m":40}}}`
	os.WriteFile(filepath.Join(dir, "t.json"), []byte(cfg), 0o600) //nolint:errcheck
	ts, err := loadTenants(filepath.Join(dir, "t.json"), filepath.Join(dir, "usage.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	up := fakeUpstream(t)
	defer up.Close()
	gw := httptest.NewServer(chain(ts.withTenant(newProxy(up.URL)), withRequestID, withLogging))
	defer gw.Close()

	post := func(key, body string) *http.Response {
		req, _ := http.NewRequest("POST", gw.URL+"/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+key)
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		io.ReadAll(resp.Body) //nolint:errcheck
		resp.Body.Close()
		return resp
	}
	if r := post("nope", `{"model":"chat"}`); r.StatusCode != 401 {
		t.Errorf("neplatný klíč: %d", r.StatusCode)
	}
	if r := post("kb", `{"model":"chat"}`); r.StatusCode != 403 {
		t.Errorf("zakázaný model: %d", r.StatusCode)
	}
	post("ka", `{"model":"chat"}`)
	post("ka", `{"model":"chat","stream":true}`)

	f, _ := os.Open(filepath.Join(dir, "usage.jsonl"))
	defer f.Close()
	var got []Usage
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var u Usage
		json.Unmarshal(sc.Bytes(), &u) //nolint:errcheck
		got = append(got, u)
	}
	if len(got) != 2 {
		t.Fatalf("kniha má %d řádků, čekám 2 (odmítnuté se neúčtují)", len(got))
	}
	if got[0].PromptTokens != 100 || got[0].CompletionTokens != 50 || got[0].CostCZK != 0.003 {
		t.Errorf("JSON usage: %+v", got[0])
	}
	if !got[1].Stream || got[1].PromptTokens != 11 || got[1].CompletionTokens != 7 {
		t.Errorf("stream usage: %+v", got[1])
	}
	rec := httptest.NewRecorder()
	ts.m.ServeHTTP(rec, nil)
	for _, want := range []string{`aigw_tokens_total{tenant="A",model="chat",kind="prompt"} 111`, `aigw_rejected_total{tenant="unknown",reason="auth"} 1`, `aigw_rejected_total{tenant="B",reason="model"} 1`} {
		if !strings.Contains(rec.Body.String(), want) {
			t.Errorf("metriky neobsahují %s\n%s", want, rec.Body.String())
		}
	}
}

// Stream musí téct průběžně i přes withLogging (regrese: Flush se zastavil na statusWriter a SSE se bufferovalo).
func TestStreamFlushThroughMiddleware(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "t.json"), []byte(`{"tenants":[{"name":"A","keys":["ka"]}]}`), 0o600) //nolint:errcheck
	ts, _ := loadTenants(filepath.Join(dir, "t.json"), filepath.Join(dir, "u.jsonl"))
	release := make(chan struct{})
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n") //nolint:errcheck
		w.(http.Flusher).Flush()
		<-release                             // druhý chunk až poté, co klient dostal první
		io.WriteString(w, "data: [DONE]\n\n") //nolint:errcheck
	}))
	defer up.Close()
	gw := httptest.NewServer(chain(ts.withTenant(newProxy(up.URL)), withRequestID, withLogging))
	defer gw.Close()
	req, _ := http.NewRequest("POST", gw.URL+"/v1/chat/completions", strings.NewReader(`{"model":"chat","stream":true}`))
	req.Header.Set("Authorization", "Bearer ka")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	close(release)
	if err != nil || !strings.HasPrefix(line, "data: {") {
		t.Fatalf("první chunk nedorazil před koncem streamu: %q %v", line, err)
	}
}
