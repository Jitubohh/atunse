# Contributing to Atunse

Thanks for helping make N-ATLaS better for every Nigerian language.

## Ways to help
- **Report a bug or idea:** open a GitHub issue with steps to reproduce, your OS and Python version.
- **Improve language detection:** add Hausa, Igbo or Yoruba function words or characters in `app/lang.py`, with example sentences in the PR description.
- **Review answers:** run Atunse, open the Review queue and fix flagged answers in a language you speak natively.
- **Code:** pick an issue, or one of the roadmap items in the README.

## Development setup
```bash
python -m pip install -r requirements.txt
ollama pull hf.co/tosinamuda/N-ATLaS-GGUF:Q4_K_M
python -m uvicorn app.main:app --reload --port 8000
```
To work without the model (UI and API only), set `ATUNSE_MOCK=1`; answers are placeholders.

## Pull requests
1. Fork, create a branch from `main` (`feature/short-name` or `fix/short-name`).
2. Keep changes focused; one feature or fix per PR.
3. Explain what changed and how you tested it. Include screenshots for UI changes.
4. Never commit `.env`, tokens, or `atunse.db` (real user data).

## Data and privacy rules
- Do not add fields that store names, emails, phone numbers or other personal data.
- Anything exported must respect the `consent` flag.
- Keep the N-ATLaS attribution in the README and UI footer.

## Code of conduct
Be respectful. Nigeria's languages and cultures are the point of this project; treat every contributor and language with care.
