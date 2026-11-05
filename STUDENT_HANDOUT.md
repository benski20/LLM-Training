# 🧠 Train Your Own Mini-LLM

## Homework: set up Kaggle (10 minutes)
1. Create a free account at **kaggle.com**.
2. Profile picture → **Settings** → **Phone verification**. Without this you get no GPU, and training is far too slow.
3. Can't verify a phone? Tell your teacher. You can pair up or use Google Colab.

## In class
1. Open the notebook link from your teacher. It opens in Kaggle.
2. Right panel → **Session options** → **Accelerator: GPU T4 x2**. Check that **Internet** is on.
3. In the **Settings** cell, type the **class code** from the board and a **nickname** (not your full name).
4. Click **Run All**. Training takes 6 minutes. Watch the dashboard, and look for yourself on the class board.
5. While it trains, read the notebook from the top and try encoding your name.

## After training
- Change the prompt and temperature in **Play with your model**.
- Try `show_guesses("To be or not to b")` and `slow_motion("ROMEO:\nI will")`.
- **Save your work**, then download `training_log.json` and replay it in the **Tiny LLM Explorer**.
- Challenge: set `DATASET = "sherlock"`, `"names"` or `"python_code"` and Run All again.

| Word | Meaning |
|---|---|
| Token | A piece of text turned into a number. |
| Parameters | The millions of numbers that hold what the model knows. |
| Loss (surprise) | How surprised the model was by the right answer. Lower is better. |
| Attention | How the model picks which earlier text matters. |
| Temperature | How risky its dice rolls are when writing. |

**Stuck?** "No working GPU found" means turning on GPU T4 x2 and clicking Run All again. Out of time? Remove the `#` before `load_pretrained()` in the last cell and run it.
