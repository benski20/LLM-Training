# 🧠 Train Your Own Mini-LLM: Student Handout

## Homework (before class): set up Kaggle (10 minutes)
1. Go to **kaggle.com** and create a free account.
2. Click your profile picture → **Settings** → **Phone verification** and verify your phone number.
   *Without this you can't use a GPU, and your training will be far too slow.*
3. Done! If you can't verify a phone, tell your teacher. You can use Google Colab instead (any Google account).

## In class
1. Open the notebook link from your teacher → click **Copy & Edit**.
2. Turn on the GPU: right-hand panel → **Session options** → **Accelerator** → **GPU T4 x2**.
3. Click **Run All** (at the top).
4. Scroll down to **🚀 Training** and watch your model learn.
5. While it trains, scroll back up and read the notebook from the start. Try encoding your own name!

## When training finishes
- **Play:** change the prompt and temperature in the *Play with your model* cells.
- **Look inside:** try `show_guesses("...")` and `slow_motion("...")` with your own text.
- **Save:** run the *Save your work* cell, and download `training_log.json` from the **Output** panel. Load it into the **Tiny LLM Explorer** → *Replay* to watch your training again.
- **Challenge:** change `DATASET` in Settings to `"sherlock"`, `"names"` or `"python_code"`, then **Run All** again.

## Words to know
| Word | Meaning |
|---|---|
| **Token** | A piece of text turned into a number. Ours are single characters. |
| **Parameters** | The millions of adjustable numbers that hold everything the model knows. |
| **Training** | Guess the next token, measure the surprise, nudge the parameters, repeat thousands of times. |
| **Loss / surprise score** | How surprised the model was by the right answers. Lower is better. |
| **Attention** | How the model decides which earlier text matters for its next guess. |
| **Temperature** | How risky the model's dice rolls are when it writes. Low = safe, high = wild. |

## Stuck?
- *"No working GPU found"*: turn on **GPU T4 x2** (step 2) and **Run All** again.
- *Training didn't finish?* Go to the very bottom, remove the `#` in front of `load_pretrained()`, and run that cell to borrow your teacher's model.
