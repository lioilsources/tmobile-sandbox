# Analytics: family size incl. childless couples, marriages per person, remarriages, top families

Module `genealogy/` (Go, SQLite). `analytics.go` holds named SQL queries in `analyticsQueries`
served to the web UI. Extend it:
- `family-size`: couples = spouses (relation `spouse`, opposite sexes) ∪ parents of a common child;
  count children per couple, **including bucket 0** (couples without any recorded child). Columns `name`, `value`.
- `top-families`: largest families by name (father, mother display names, their ids `otec_id`/`matka_id`,
  `value` = number of distinct children), top 15.
- `marriages-per-person`: how many marriage records each person appears in as groom/bride (`zenich`/`nevesta`),
  grouped by count with columns `muzi`, `zeny`.
- `remarriages`: per decade (`name`), total marriages (`snatky`), widowed grooms (`vdovci`, marital_status `vdovec`)
  and widowed brides (`vdovy`, `vdova`).

Every analytics query must run on the test fixture; bucket 0 of family-size must contain the
newly-weds of 1901/1905 from the fixture.

Verify: `cd genealogy && go test .` (TestAnalyticsQueries).
