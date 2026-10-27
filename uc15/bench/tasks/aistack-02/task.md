# gen-queue portal proxy: reject path traversal in /cpe/workflow/save

`services/gen-queue/internal/portal` proxies requests to a ComfyUI portal endpoint whose
`/cpe/workflow/save` joins the user-supplied `name` into a path without checking (path traversal).

In `(*Proxy).ServeHTTP`, before proxying a request to `/cpe/workflow/save`:
- read the JSON body (cap it, e.g. 16 MiB → 413 when larger), invalid JSON → 400;
- if `name` is present it must be a bare file name: non-empty, no `/`, `\`, NUL, and not starting with `.`
  → otherwise 400 and the request is not forwarded;
- valid requests are forwarded with the original body intact (restore `Body` and `ContentLength`).

Verify: `cd services/gen-queue && go test ./internal/portal` (TestSaveName).
