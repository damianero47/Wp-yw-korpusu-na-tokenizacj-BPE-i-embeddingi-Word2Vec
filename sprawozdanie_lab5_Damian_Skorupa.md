# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy

**Zajęcia nr / temat:** 5 — Wpływ korpusu na tokenizację BPE i embeddingi Word2Vec

**Autor / grupa:** Damian Skorupa / GL03

**Data:** 12.11.2026

## 1. Cel ćwiczenia

Na własnym korpusie zbadano wpływ rozmiaru słownika BPE na podział form fleksyjnych oraz wpływ kontekstów wyrazu dwuznacznego na jego sąsiadów w Word2Vec. Sprawdzono też, czy w embeddingach widać grupy tematyczne. Badany wyraz: **zamek** (1 — budowla, 2 — mechanizm do zamykania).

## 2. Metoda i implementacja

**Narzędzia:** `tokenizers` (BPE, `Whitespace`), `gensim` (Word2Vec), `scikit-learn` (PCA), `numpy`, `matplotlib`. 

**Korpus:** 60 różnych zdań ułożonych na potrzeby zadania, 3 pule po 20: *budowla* i *mechanizm* (forma „zamek” raz w każdym zdaniu) oraz *bez badanego wyrazu*. Razem 419 słów, 242 różne. Przygotowanie tekstu: małe litery, usunięcie interpunkcji, podział po spacjach.

**Próbka BPE (8 par widziana → niewidziana):** klucz → kluczami, wieża → wieżami, król → królowi, drzwi → drzwiami, rycerze → rycerzom, ślusarz → ślusarzowi, zamek → zamkami, turyści → turystami. Obecność form sprawdzono po normalizacji.

**Ustawienia:**

| Eksperyment | Parametry |
|---|---|
| BPE | cały korpus, `vocab_size` ∈ {100, 300, 1000}, `min_frequency=1`, `[UNK]` |
| Word2Vec | wariant 1: 20 zdań bez wyrazu + 12 budowla + 4 mechanizm; wariant 2: 20 + 4 + 12 (po 36 zdań i 16 × „zamek”), powielenie ×5; `vector_size=50`, `window=3`, `min_count=2`, `sg=1`, `epochs=200`, `seed=0` |
| Grupy (wybrane przed treningiem) | budowla: król, wieża, mury, rycerze, turyści; mechanizm: klucz, drzwi, ślusarz, kłódka, sejf (każde słowo ≥ 2 razy w wariancie 1) |

```python
def trenuj_bpe(vocab_size):
    """Trenuje tokenizer BPE na całym korpusie; zmienia się tylko limit."""
    tk = Tokenizer(models.BPE(unk_token='[UNK]'))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tr = trainers.BpeTrainer(vocab_size=vocab_size,
                             special_tokens=['[UNK]'], min_frequency=1)
    tk.train_from_iterator(KORPUS_BPE, tr)
    return tk
```

## 3. Wyniki

**Hipoteza BPE:** większy słownik skróci podział form widzianych (do 1 tokenu), a dla niewidzianych spadek będzie mały.

| Limit | Faktyczny \|V\| | Widziane: śr. tokenów | [UNK] | Niewidziane: śr. tokenów | [UNK] |
|---|---|---|---|---|---|
| 100 | 100 | 1,50 | 0 | 3,75 | 0 |
| 300 | 300 | 1,00 | 0 | 3,50 | 0 |
| 1000 | 739 | 1,00 | 0 | 3,50 | 0 |

| Forma niewidziana | limit 100 | limit 300 | limit 1000 |
|---|---|---|---|
| ślusarzowi | ślu · sarz · ow · i | ślusarz · ow · i | ślusarz · ow · i |
| zamkami | zam · ka · m · i | zam · ka · mi | zam · ka · mi |

**Word2Vec — 5 najbliższych sąsiadów „zamek”:**

| Poz. | Wariant 1 (12 × budowla) | cos | Wariant 2 (12 × mechanizm) | cos |
|---|---|---|---|---|
| 1 | wróg | 0,38 | ślusarz | 0,51 |
| 2 | lata | 0,37 | klucza | 0,50 |
| 3 | okolicą | 0,37 | bez | 0,49 |
| 4 | drzwiach | 0,37 | ten | 0,46 |
| 5 | mostu | 0,37 | drzwiach | 0,46 |

**PCA (wariant 1) i średni cosinus:**

<img width="764" height="563" alt="wykres1" src="https://github.com/user-attachments/assets/72deebb9-c340-4ba2-b63a-8584600bbfef" />


| Pary | Liczba par | Średni cos |
|---|---|---|
| wewnątrz grupy *budowla* | 10 | 0,240 |
| wewnątrz grupy *mechanizm* | 10 | 0,462 |
| między grupami | 25 | 0,213 |

**Baseline:** tokenizacja znakowa daje tyle tokenów, ile liter (np. „ślusarzowi” = 10), a słowna zamienia każdą niewidzianą formę na `[UNK]`.

## 4. Analiza i interpretacja

- **BPE:** hipoteza się potwierdziła. Formy widziane spadają do 1 tokenu już przy 300 wpisach. Niewidziane zyskują niewiele (3,75 → 3,50), a powyżej 300 nic się nie zmienia, bo nowe wpisy to całe słowa korpusu.
  - W *ślusarzowi* łączy się rdzeń `ślusarz` z formy treningowej *ślusarz*.
  - W *zamkami* powstaje `mi` (z *wymienił*, *miasta*).
  - Końcówki *-ami*, *-owi* nie powstały, bo są w korpusie rzadkie.
  - Przypadek błędny: *kluczami* → `klu · c · zam · i`. Wcześniej wyuczona reguła `za + m` (z częstego „zamek”) przecina rdzeń *klucz*.
- **Word2Vec:** ranking się odwrócił. W wariancie 1 sąsiedzi pochodzą głównie ze zdań o budowli („Wróg oblegał zamek…”, „Turyści fotografują zamek z mostu”), w wariancie 2 ze zdań o mechanizmie („Ślusarz otworzył zamek bez klucza”).
  - *wróg*, *mostu*, *klucza*, *bez* nie mają wektora w drugim modelu, bo ich jedyne zdanie nie weszło do drugiego wariantu.
  - *ślusarz* awansuje z 46. na 1. miejsce.
  - Niskie wartości cos pokazują, że statyczny wektor „zamek” miesza oba znaczenia.
- **Grupy:** na wykresie grupa *mechanizm* tworzy zwarte skupisko, a *budowla* jest rozproszona; „zamek” leży bliżej słów o budowli. Cosinusy to potwierdzają: 0,46 dla *mechanizm* wobec 0,21 między grupami, ale tylko 0,24 dla *budowla*.
  - Przykład: *klucz–ślusarz* (0,66) występują razem („Ślusarz dorobił klucz do piwnicy”).
  - *mury–wieża* (0,17) nigdy nie występują w jednym zdaniu. Spójność grupy zależy więc od tego, czy słowa z grupy występują razem, a nie od liczby zdań na dany temat.
- **Ograniczenia:** mały korpus, PCA wyjaśnia tylko część wariancji, wyniki zależą od ziarna losowego.

## 5. Wnioski

1. Większy słownik BPE pomaga głównie formom widzianym. Dla nowych form fleksyjnych zysk jest mały i szybko się kończy.
2. Proporcja zdań z danym znaczeniem decyduje o sąsiadach wyrazu dwuznacznego w Word2Vec.
3. Grupy tematyczne są widoczne tylko wtedy, gdy słowa z grupy występują razem w zdaniach.

**Kierunki poprawy:** większy korpus bez powielania, FastText dla form niewidzianych, embeddingi kontekstowe (np. HerBERT) dla wyrazów wieloznacznych.
