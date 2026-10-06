"""Génération de rapport via l'API Groq — gratuite, sans carte bancaire,
14 400 requêtes/jour. Remplace Claude comme moteur d'interprétation par
défaut : la qualité (gpt-oss-120b, 120 milliards de paramètres) dépasse
largement le modèle local (Qwen2.5-7B, voir local_llm.py), sans dépendre
d'une clé payante.

gpt-oss-120b (modèle ouvert d'OpenAI, hébergé par Groq) est choisi après
vérification live de client.models.list() sur ce compte : Kimi K2 et les
Llama 3.x/4 remontés par la recherche web n'y sont pas disponibles. Étant
un modèle OpenAI, il supporte nativement les structured outputs (JSON
schema strict), condition nécessaire pour remplacer Claude proprement.

Clé gratuite : https://console.groq.com/keys (aucune carte bancaire), puis
`export GROQ_API_KEY=...` avant de lancer le pipeline.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Chargé ici (plutôt que dans chaque script d'entrée) car c'est le seul
# module que les trois chemins (CLI, webapp.py, llm_worker.py/precall_worker.py)
# importent systématiquement avant de lire GROQ_API_KEY.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = "openai/gpt-oss-120b"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY manquante. Clé gratuite (aucune carte bancaire) "
                "sur https://console.groq.com/keys, puis : export GROQ_API_KEY=..."
            )
        _client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    return _client


def _enforce_strict(node: object, defs: dict) -> None:
    """Le mode structured output strict de Groq/OpenAI exige
    additionalProperties=False et tous les champs dans required, partout
    dans le schéma — Pydantic ne le fait pas nativement pour les modèles
    imbriqués. On le force récursivement sur tout le schéma généré."""
    if not isinstance(node, dict):
        return
    if "properties" in node:
        node["additionalProperties"] = False
        node["required"] = list(node["properties"].keys())
        for child in node["properties"].values():
            _enforce_strict(child, defs)
    if "items" in node:
        _enforce_strict(node["items"], defs)
    if "$ref" in node:
        ref_name = node["$ref"].split("/")[-1]
        _enforce_strict(defs.get(ref_name), defs)
    for key in ("anyOf", "oneOf", "allOf"):
        for child in node.get(key, []):
            _enforce_strict(child, defs)


def _strict_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    for definition in defs.values():
        _enforce_strict(definition, defs)
    _enforce_strict(schema, defs)
    return schema


def parse(system: str, user: str, output_format: type[BaseModel], max_tokens: int = 16000) -> BaseModel:
    client = _get_client()
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": output_format.__name__,
            "schema": _strict_schema(output_format),
            "strict": True,
        },
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    # gpt-oss-120b rate un JSON valide de façon occasionnelle et non
    # déterministe même en mode schema strict (observé en test : échec puis
    # succès immédiat sur un appel identique) — une retentative absorbe ce
    # raté avant de céder la place au modèle local, nettement moins bon.
    last_error: Exception | None = None
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL, max_tokens=max_tokens, messages=messages, response_format=schema,
                reasoning_effort="low",  # moins de tokens de réflexion = rapport complet dans le budget gratuit
            )
            return output_format.model_validate_json(response.choices[0].message.content)
        except Exception as e:
            last_error = e
    raise last_error
