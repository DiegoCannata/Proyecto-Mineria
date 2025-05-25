def predecir_emocion_probs_robertuito(text, labels=EMO_LABELS):
  """Devuelve (label, [probs…]) para cada emoción del transformer robertuito."""
    res = analyzer.predict(text)
    # Mapa label -> probabilidad
    score_map = {k: v for k, v in res.probas.items()}
    # Etiqueta con mayor probabilidad
    top_label = max(score_map.items(), key=lambda x: x[1])[0]
    # Vector de probabilidades en orden
    probs = [score_map.get(lab, 0.0) for lab in labels]
    return top_label, probs

def predecir_emocion_probs_transformers(text, classifier, labels=EMO_LABELS):
  """Devuelve (label, [probs…]) para cada emoción."""
    salida = classifier(text)[0]
    # construye un mapa label->score, ignorando 'others'
    score_map = {d['label'].lower(): d['score'] for d in salida if d['label'].lower() != 'others'}
    # etiqueta con mayor probabilidad
    top_label = max(score_map.items(), key=lambda x: x[1])[0]

    # extrae probabilidades en el orden definido por labels
    probs = [score_map.get(lab, 0.0) for lab in labels]

    return top_label, probs

def get_cls_embedding(text):
  """Extrae embedding del token [CLS] de un transformer."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[0, 0, :]
    return cls_emb.numpy()
