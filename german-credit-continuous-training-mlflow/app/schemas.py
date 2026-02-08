from pydantic import BaseModel, Field

class CreditInput(BaseModel):
    duration: int = Field(..., ge=1, le=72)           # meses
    credit_amount: float = Field(..., ge=0)           # USD
    age: int = Field(..., ge=18, le=100)

    checking_status: str
    employment: str
    savings_status: str
    purpose: str

class PredictRequest(CreditInput):
    # ID "de presentación": sirve para identificar el caso en la app y poder registrar el resultado real más adelante.
    client_id: str | None = Field(default=None, min_length=1, max_length=60)

class PredictResponse(BaseModel):
    model_version: str
    proba_good: float
    decision: str  # APROBADO / RECHAZADO

class OutcomeFeedback(BaseModel):
    client_id: str = Field(..., min_length=1, max_length=60)
    paid: bool  # True=Pagó (BUENO), False=No pagó (MALO)

class FeedbackItem(BaseModel):
    # Compatibilidad: si alguien envía el feedback con los datos completos.
    client_id: str | None = Field(default=None, min_length=1, max_length=60)
    x: CreditInput
    target: int  # 1=BUENO (pagó), 0=MALO (no pagó)
