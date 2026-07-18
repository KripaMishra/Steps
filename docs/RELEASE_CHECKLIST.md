# Public release checklist

- [ ] `make test`, `make lint`, and `make evaluate` pass.
- [ ] `docker compose config` and `docker compose --profile full config` pass when Docker is available.
- [ ] The three-question [demo checklist](DEMO_CHECKLIST.md) passes.
- [ ] `git ls-files` contains no secrets, caches, locks, crawl output, runtime results, or database volumes.
- [ ] Fixture changes are original summaries with current source/provenance fields and remain under 1 MB.
- [ ] No NVIDIA material is described as relicensed.
- [ ] No unsupported accuracy, latency, cost, or production-readiness claim was added.
- [ ] Dependency and container versions have been reviewed.
- [ ] Security and contribution links work.
