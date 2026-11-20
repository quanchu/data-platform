"""Deterministic shared model doubles; no provider calls or model downloads."""

import json
from pydantic_ai.embeddings.test import TestEmbeddingModel
from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from data_platform.digest.openie import (
    BackendMetadata,
    ExtractionResponse,
    Proposition,
    ResolutionResponse,
    ResolvedSentence,
)

META = BackendMetadata(backend="fixture")


def models(overrides=None):
    overrides = overrides or {}
    calls = []

    def respond(kind):
        def inner(messages, info):
            tool = info.output_tools[0]
            schema = tool.parameters_json_schema
            fields = schema["properties"]
            payload = next(
                json.loads(p.content)
                for m in reversed(messages)
                for p in m.parts
                if isinstance(p, UserPromptPart)
            )
            calls.append((kind, tuple(fields), payload))
            if kind == "generation":
                assert set(fields) in ({"text"}, {"claims"}), fields
                value = (
                    {"text": payload["target_sentence"].strip()}
                    if "text" in fields
                    else {"claims": [payload["clarified_text"].strip()]}
                )
            else:
                assert "text" not in fields and "claims" not in fields, fields
                if "verdict" in fields:
                    value = {"verdict": "fully_verifiable"}
                elif "referential" in fields:
                    value = {
                        "referential": "absent",
                        "structural": "absent",
                        "expand_names": False,
                    }
                else:
                    value = {key: True for key in fields}
            for key in list(value):
                if key in overrides:
                    setting = overrides[key]
                    value[key] = setting(payload) if callable(setting) else setting
            return ModelResponse(
                parts=[ToolCallPart(tool.name, value)], model_name=kind
            )

        return inner

    return (
        FunctionModel(respond("decision")),
        FunctionModel(respond("generation")),
        calls,
    )


class Embeddings(TestEmbeddingModel):
    def __init__(self, name="deterministic"):
        super().__init__(model_name=name)
        self.calls = []
        self.fail = False

    async def embed(self, inputs, *, input_type, settings=None):
        if self.fail:
            raise RuntimeError("embedding failed")
        response = await super().embed(inputs, input_type=input_type, settings=settings)
        self.calls.append((list(response.inputs), input_type))
        response.embeddings = [
            [1.0, 0.0] if "revenue" in text else [0.0, 1.0] for text in response.inputs
        ]
        return response


class Resolver:
    def __init__(self):
        self.sections = []

    async def resolve(self, *, title, section):
        self.sections.append(section)
        return ResolutionResponse(
            sentences=[
                ResolvedSentence(
                    sentence_index=s.index, text=s.text, status="unchanged"
                )
                for s in section.sentences
            ],
            metadata=META,
        )


class Extractor:
    def __init__(self):
        self.inputs = []

    async def extract(self, *, sentence, resolved):
        self.inputs.append((sentence, resolved))
        return ExtractionResponse(
            propositions=[
                Proposition(
                    subject="Fixture",
                    relation="states",
                    object=resolved.text,
                    text=resolved.text,
                )
            ],
            metadata=META,
        )
