from collections import defaultdict
import string
import random
import numpy as np

rep_map = defaultdict(lambda: "*")
rep_map.update({
    "i": "!", "s": "$", "o": "0", "a": "@",
})
rep_map.update({
    str(i): str(i) for i in range(10)
})

noise_rate = 0.5
targeted_replace_rate = 0.1
untargeted_replace_rate = 0.3
insert_rate = 0.3

alphabet = list(string.ascii_lowercase)

def random_perturbation(s: str):
    if len(s) < 4: return s
    if random.random() <= noise_rate:
        rep_idx = np.random.randint(len(s) - 2) + 1 # exclude first and final positions
        r = random.random()
        if r <= targeted_replace_rate: # replace
            return "".join([rep_map[c] if i == rep_idx else c for (i, c) in enumerate(s)])
        elif r <= targeted_replace_rate + untargeted_replace_rate:
            return "".join([np.random.choice(alphabet) if i == rep_idx else c for (i, c) in enumerate(s)])
        elif r <= targeted_replace_rate + untargeted_replace_rate + insert_rate:
            return s[:rep_idx] + np.random.choice(alphabet) + s[rep_idx:]
        else: # remove
            return s[:rep_idx] + s[rep_idx + 1:]
    else:
        return s


def swap(word):
  cword = word
  if len(word) >= 4:
    s = np.random.randint(1, len(word) - 2)
    cword = word[:s] + word[s + 1] + word[s] + word[s + 2:]
  return (cword)


def remove(word):
  s = np.random.randint(0, len(word))
  if len(word) > 2:
    cword = word[:s] + word[s + 1:]
  else:
    cword = word
  return cword


def insert(word):
  cword = word
  s = np.random.randint(0, len(word) + 1)
  cword = word[:s] + chr(97 + np.random.randint(0, 26)) + word[s:]

  return (cword)


def homoglyph(word):
  s = np.random.randint(0, len(word))
  homos = {'-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l',
           '0': 'O', "'": '`', 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ',
           'i': 'і', 'j': 'ϳ', 'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ',
           's': 'ѕ', 't': '𝚝', 'u': 'ս', 'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'}

  if word[s] in homos:
    rletter = homos[word[s]]
  else:
    rletter = word[s]
  cword = word[:s] + rletter + word[s + 1:]

  return (cword)


def repeat_char(word):
  s = np.random.randint(0, len(word))
  rletter = word[s]
  cword = word[:s] + rletter + word[s:]

  return (cword)


def distractor(word):
  s = np.random.randint(0, len(topClass1))
  distractor_word = topClass1[s][1]
  cword = word + ' ' + distractor_word

  return (cword)