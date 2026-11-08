// comfy_relay: token-checking reverse proxy in front of ComfyUI (on the L40S VM, exposed via tailscale serve).
//
// Only the ComfyUI endpoints PartyCockpit needs are forwarded; every request must carry
// `Authorization: Bearer <token>`, where the token is read from RELAY_TOKEN_FILE.
package main

import (
	"crypto/subtle"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"time"
)

var allowed = []string{"/prompt", "/history", "/view", "/upload/image", "/system_stats"}

func pathAllowed(p string) bool {
	for _, a := range allowed {
		if p == a || strings.HasPrefix(p, a+"/") {
			return true
		}
	}
	return false
}

func authorized(h, token string) bool {
	got, ok := strings.CutPrefix(h, "Bearer ")
	return ok && token != "" && subtle.ConstantTimeCompare([]byte(got), []byte(token)) == 1
}

func handler(upstream *url.URL, token string, maxBody int64) http.Handler {
	rp := httputil.NewSingleHostReverseProxy(upstream)
	rp.Director = func(r *http.Request) {
		r.URL.Scheme, r.URL.Host, r.Host = upstream.Scheme, upstream.Host, upstream.Host
		r.Header.Del("Authorization")
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !authorized(r.Header.Get("Authorization"), token) {
			slog.Warn("unauthorized", "remote", r.RemoteAddr, "path", r.URL.Path)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if !pathAllowed(r.URL.Path) {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if r.ContentLength > maxBody {
			http.Error(w, "too large", http.StatusRequestEntityTooLarge)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxBody)
		slog.Info("proxy", "remote", r.RemoteAddr, "method", r.Method, "path", r.URL.Path)
		rp.ServeHTTP(w, r)
	})
}

func main() {
	listen := flag.String("listen", ":18188", "listen address")
	up := flag.String("upstream", "http://127.0.0.1:8188", "ComfyUI base URL")
	tokenFile := flag.String("token-file", os.Getenv("RELAY_TOKEN_FILE"), "file with the bearer token")
	maxMB := flag.Int64("max-mb", 32, "max request body in MB")
	flag.Parse()
	b, err := os.ReadFile(*tokenFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "token file:", err)
		os.Exit(1)
	}
	token := strings.TrimSpace(string(b))
	if len(token) < 24 {
		fmt.Fprintln(os.Stderr, "token too short")
		os.Exit(1)
	}
	u, err := url.Parse(*up)
	if err != nil {
		fmt.Fprintln(os.Stderr, "upstream:", err)
		os.Exit(1)
	}
	srv := &http.Server{Addr: *listen, Handler: handler(u, token, *maxMB<<20), ReadHeaderTimeout: 10 * time.Second}
	slog.Info("comfy_relay", "listen", *listen, "upstream", *up)
	if err := srv.ListenAndServe(); err != nil {
		slog.Error("serve", "err", err)
		os.Exit(1)
	}
}
