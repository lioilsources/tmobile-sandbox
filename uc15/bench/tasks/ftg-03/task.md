# Serve static style-map packs under GET /stylemaps/{path...}

The app's style picker needs static packs (built elsewhere and copied into `<data>/stylemaps`):
`index.json`, `<set>/map.json`, `<set>/atlas.webp`, `<set>/t/<i>.webp`.

In FineTuneGallery `server/` add a handler `func (s *server) handleStylemap(w, r)` registered as
`GET /stylemaps/{path...}` (and create the `stylemaps` data subdirectory at startup):
- serve files only — no directory listings (404), no dotfiles/dot-directories (404), nothing outside
  `<dataDir>/stylemaps` (also for encoded `..`) → 404;
- `.json` files get `Cache-Control: no-cache`, everything else `public, max-age=86400`;
- keep `/stylemaps/` requests out of the request log like `/img/`.

Verify: `cd server && go test .` (TestStylemapServesPackFiles).
