# Submission requirements (from official ICLR 2026 Author Guide; verify against 2027 guide when live)

Source: https://iclr.cc/Conferences/2026/AuthorGuide (fetched 2026-07-18)

## Format

- **Main text: 9 pages or fewer at submission** (strictly enforced); 10 pages allowed at discussion/camera-ready.
- References: unlimited, do not count toward page limit.
- Appendices: unlimited, placed after bibliography; reviewers not obligated to read them.
- Official ICLR LaTeX style required (`iclr20XX_conference.sty`).

## Anonymization (double-blind)

- Author identity revealed anywhere in main text or supplementary => **desk rejection**.
- Citing your own arXiv papers is fine in third person.

## Supplementary material

- Single combined file or separate docs; source code may be uploaded as supplementary.
- Same deadline as full paper.

## Statements

- **Reproducibility Statement**: strongly encouraged, end of main text before references, does not count toward page limit.
- **Ethics Statement**: optional, <=1 page, does not count; recommended for human subjects / dataset releases / bias.

## Policies with desk-rejection teeth

1. Identity revelation.
2. Missing qualified reviewer registration (**reciprocal reviewing**: every submission needs >=1 author
   registered to review >=3 papers; authors on 3+ submissions must review 6).
3. Placeholder abstracts at abstract deadline.
4. Exceeding page limits.
5. **Non-disclosure of significant LLM contribution** — if LLMs played a significant role in ideation
   and/or writing to the extent they could be regarded as a contributor, the precise role must be described.
   (Directly relevant to this pipeline: the final paper MUST carry an LLM-usage disclosure, and per
   orchestrator integrity rule #6 a human reviews everything before submission.)

## Checklist for our run

- [ ] Re-verify 2027 deadlines + template when iclr.cc/2027 goes live
- [ ] LLM-usage disclosure paragraph drafted in Stage 4
- [ ] Reproducibility statement fed from Stage 3 artifacts (seeds, configs, commit hashes)
- [ ] At least one author registered as reciprocal reviewer (user's responsibility; flag before submission)
