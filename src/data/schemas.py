"""
Schemas Pydantic de entrada e saída para o pipeline de crédito.

Responsabilidades:
- Validar dados de entrada antes de chegarem ao modelo
- Documentar contratos de dados da API
- Converter entre formatos (dict ↔ DataFrame)
"""

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class CreditClientInput(BaseModel):
    """
    Schema de entrada para predição de risco de crédito.

    Baseado no dataset UCI Credit Card com features derivadas.
    Todos os campos refletem um único cliente no momento da avaliação.
    """

    limit_balance: float = Field(
        gt=0,
        description="Limite de crédito em dólares de Taiwan (NT$).",
        examples=[50000.0],
    )
    age: int = Field(
        ge=18,
        le=100,
        description="Idade do cliente em anos.",
        examples=[35],
    )
    sex: int = Field(
        ge=1,
        le=2,
        description="Sexo do cliente: 1=masculino, 2=feminino.",
        examples=[2],
    )
    education: int = Field(
        ge=1,
        le=6,
        description=(
            "Nível de escolaridade: 1=pós-graduação, 2=graduação, "
            "3=ensino médio, 4=outros, 5=desconhecido, 6=desconhecido."
        ),
        examples=[2],
    )
    marriage: int = Field(
        ge=0,
        le=3,
        description="Estado civil: 0=desconhecido, 1=casado, 2=solteiro, 3=outros.",
        examples=[1],
    )
    pay_history: list[int] = Field(
        min_length=6,
        max_length=6,
        description=(
            "Histórico de pagamento dos últimos 6 meses, alinhado ao UCI: "
            "PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 (nesta ordem). "
            "Valores: -2=sem consumo, -1=pago em dia, 0=uso do crédito rotativo, "
            "1-9=meses de atraso."
        ),
        examples=[[-1, -1, 0, 0, 1, 0]],
    )
    bill_amounts: list[float] = Field(
        min_length=6,
        max_length=6,
        description="Valor das faturas nos últimos 6 meses (BILL_AMT1 a BILL_AMT6) em NT$.",
        examples=[[23000.0, 18500.0, 21000.0, 15000.0, 17000.0, 20000.0]],
    )
    pay_amounts: list[float] = Field(
        min_length=6,
        max_length=6,
        description="Valor dos pagamentos nos últimos 6 meses (PAY_AMT1 a PAY_AMT6) em NT$.",
        examples=[[1500.0, 2000.0, 800.0, 1000.0, 500.0, 1200.0]],
    )

    @field_validator("pay_history")
    @classmethod
    def validate_pay_history_values(cls, values: list[int]) -> list[int]:
        """Valida que os valores de histórico de pagamento estão no range esperado."""
        valid_range = range(-2, 10)
        invalid = [v for v in values if v not in valid_range]
        if invalid:
            raise ValueError(
                f"Valores inválidos em pay_history: {invalid}. "
                f"Esperado: inteiros entre -2 e 9."
            )
        return values

    @model_validator(mode="after")
    def validate_pay_amounts_not_exceed_bill(self) -> "CreditClientInput":
        """Emite aviso se pagamentos excedem faturas consistentemente (possível erro de dados)."""
        overpayments = sum(
            1
            for pay, bill in zip(self.pay_amounts, self.bill_amounts, strict=False)
            if pay > bill * 2 and bill > 0
        )
        if overpayments > 3:
            raise ValueError(
                "Mais de 3 meses com pagamentos que excedem o dobro da fatura. "
                "Verifique se os dados estão corretos."
            )
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Converte o schema para DataFrame compatível com o pipeline de features."""
        row: dict = {
            "LIMIT_BAL": self.limit_balance,
            "AGE": self.age,
            "SEX": self.sex,
            "EDUCATION": self.education,
            "MARRIAGE": self.marriage,
        }
        # UCI Credit Card: colunas PAY_0, PAY_2..PAY_6 (não existe PAY_1 no dataset original)
        pay_uci_names = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        for name, pay, bill, pay_amt in zip(
            pay_uci_names,
            self.pay_history,
            self.bill_amounts,
            self.pay_amounts,
            strict=True,
        ):
            row[name] = pay
        for i, (bill, pay_amt) in enumerate(
            zip(self.bill_amounts, self.pay_amounts, strict=True),
            start=1,
        ):
            row[f"BILL_AMT{i}"] = bill
            row[f"PAY_AMT{i}"] = pay_amt
        return pd.DataFrame([row])


class PredictionOutput(BaseModel):
    """Schema de saída de uma predição de risco de crédito."""

    client_id: str = Field(description="Identificador único gerado para a requisição.")
    default_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidade prevista de inadimplência.",
    )
    decision: str = Field(
        description="Decisão de crédito: 'APROVADO' ou 'NEGADO'.",
    )
    threshold_used: float = Field(
        description="Threshold utilizado para a decisão.",
    )
    top_factors: list[dict[str, str | float]] = Field(
        description="Top 5 features e seus valores SHAP para esta predição.",
    )
    model_version: str = Field(
        description="Versão do modelo utilizado.",
    )
    timestamp: str = Field(
        description="ISO 8601 timestamp da predição.",
    )
