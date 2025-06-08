
from pydantic import BaseModel

class VoterRegistration(BaseModel):
    name: str
    email: str
    image_data: str  # Base64 encoded image

class FaceAuthentication(BaseModel):
    image_data: str  # Base64 encoded image

class VoteRequest(BaseModel):
    voter_name: str
    candidate_id: str

class BlockchainConfig(BaseModel):
    contract_address: str
    rpc_url: str
    private_key: str
    account_address: str
