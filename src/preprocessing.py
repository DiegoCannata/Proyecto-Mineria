def normalizar_texto(text):
  """
    - Minusculiza, elimina acentos y dígitos
    - Quita repetición excesiva de letras
    - Corrección ortográfica con pyspellchecker
    - Lematiza y filtra stopwords en español
  """
    # Lowercase
    text = text.lower()

    # Elimina saltos de línea
    text = text.replace('\n', ' ')

    # Normaliza caracteres Unicode y elimina acentos o tildes
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = unidecode(text)

    # Elimina las repeticiones excesivas de una misma letra. Ejemplo: 'holaaa' = 'hola'
    text = re.sub(r'(.)\1+', r'\1', text)

    # Elimina dígitos
    text = re.sub(r'\d+', ' ', text)

    # Elimina caracteres especiales y puntuación
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)

    # Colapsa espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return ''

    # Corrección ortográfica
    tokens = text.split()
    misspelled = spell.unknown(tokens)
    corrected = [spell.correction(tok) or tok for tok in tokens]
    corrected_text = ' '.join(corrected)

    # 8. Lematización y filtrado de stopwords
    doc = nlp(corrected_text)
    lemmas = []
    for token in doc:
        if token.is_alpha:
            lemma = token.lemma_
            if lemma not in stop_es or lemma in keep_words:
                lemmas.append(lemma)

    return ' '.join(lemmas)
