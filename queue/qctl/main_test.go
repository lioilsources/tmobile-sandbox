package main

import (
	"testing"
	"time"
)

func TestSortQueueShortFirstFillerLast(t *testing.T) {
	t0 := time.Date(2026, 11, 6, 10, 0, 0, 0, time.UTC)
	js := []*Job{
		{Name: "az", EstMin: 600, Filler: true, Submitted: t0},
		{Name: "train", EstMin: 240, Submitted: t0},
		{Name: "bench-b", EstMin: 90, Submitted: t0.Add(time.Minute)},
		{Name: "bench-a", EstMin: 90, Submitted: t0},
		{Name: "tiny-filler", EstMin: 5, Filler: true, Submitted: t0},
	}
	sortQueue(js)
	want := []string{"bench-a", "bench-b", "train", "tiny-filler", "az"}
	for i, n := range want {
		if js[i].Name != n {
			t.Fatalf("pos %d: got %s want %s", i, js[i].Name, n)
		}
	}
}

func TestValidate(t *testing.T) {
	w := &Job{Name: "w", Kind: "window", EstMin: 90, Ready: "R", Done: "D"}
	if err := validate(w); err != nil || w.GuardMin != 180 || w.Profile != "off" {
		t.Fatalf("window defaults: %v %+v", err, w)
	}
	bad := []*Job{
		{Name: "a b", Kind: "cmd", Cmd: "x", EstMin: 1},
		{Name: "c", Kind: "cmd", EstMin: 1},
		{Name: "d", Kind: "cmd", Cmd: "x"},
		{Name: "e", Kind: "window", EstMin: 1, Ready: "R", Done: "D", Filler: true},
		{Name: "f", Kind: "other", EstMin: 1},
	}
	for _, j := range bad {
		if validate(j) == nil {
			t.Fatalf("%s should be invalid", j.Name)
		}
	}
}
