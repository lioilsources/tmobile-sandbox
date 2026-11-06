package main

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestRelay(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Error("token leaked upstream")
		}
		w.Write([]byte("ok " + r.URL.Path))
	}))
	defer up.Close()
	u, _ := url.Parse(up.URL)
	tok := "abcdefghijklmnopqrstuvwxyz0123"
	h := handler(u, tok, 1<<20)
	cases := []struct {
		auth, path string
		want       int
	}{
		{"", "/system_stats", 401},
		{"Bearer wrong", "/system_stats", 401},
		{"Bearer " + tok, "/system_stats", 200},
		{"Bearer " + tok, "/history/123", 200},
		{"Bearer " + tok, "/api/userdata", 404},
		{"Bearer " + tok, "/prompts", 404},
	}
	for _, c := range cases {
		r := httptest.NewRequest("GET", c.path, nil)
		if c.auth != "" {
			r.Header.Set("Authorization", c.auth)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != c.want {
			t.Errorf("%s %q: got %d want %d", c.path, c.auth, w.Code, c.want)
		}
	}
	r := httptest.NewRequest("POST", "/upload/image", strings.NewReader(strings.Repeat("x", 2<<20)))
	r.Header.Set("Authorization", "Bearer "+tok)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code < 400 {
		t.Errorf("oversized body passed: %d", w.Code)
	}
}
