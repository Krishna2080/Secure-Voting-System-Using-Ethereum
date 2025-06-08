
import os
import pickle
from typing import Dict, Set, Any
# blockchain_service import kept for vote casting, not for auth
from blockchain_service import blockchain_service

# File paths
face_data_dir = "face_embeddings"
voted_users_file = "voted_users.pkl"
face_registry_file = "face_registry.pkl"

if not os.path.exists(face_data_dir):
    os.makedirs(face_data_dir)

class VoterService:
    def __init__(self):
        self.voted_users = self._load_voted_users()
        self.face_registry = self._load_face_registry()

    def _load_voted_users(self) -> Set[str]:
        if os.path.exists(voted_users_file):
            with open(voted_users_file, "rb") as file:
                return pickle.load(file)
        return set()

    def _load_face_registry(self) -> Dict[str, Any]:
        if os.path.exists(face_registry_file):
            with open(face_registry_file, "rb") as file:
                return pickle.load(file)
        return {}

    def _save_voted_users(self):
        with open(voted_users_file, "wb") as file:
            pickle.dump(self.voted_users, file)

    def _save_face_registry(self):
        with open(face_registry_file, "wb") as file:
            pickle.dump(self.face_registry, file)

    def register_voter(self, name: str, embedding: list) -> bool:
        user_face_file = os.path.join(face_data_dir, f"{name}.pkl")
        with open(user_face_file, "wb") as file:
            pickle.dump([embedding], file)

        self.face_registry[name] = embedding
        self._save_face_registry()
        return True

    def is_voter_registered(self, name: str) -> bool:
        return name in self.face_registry

    def load_user_embeddings(self) -> Dict[str, Any]:
        user_embeddings = {}
        for filename in os.listdir(face_data_dir):
            if filename.endswith('.pkl'):
                voter_name = filename.replace(".pkl", "")
                with open(os.path.join(face_data_dir, filename), "rb") as file:
                    user_embeddings[voter_name] = pickle.load(file)
        return user_embeddings

    def has_voted_locally(self, voter_name: str) -> bool:
        return voter_name in self.voted_users

    def has_voted(self, voter_name: str) -> bool:
        # 🚫 Removed blockchain check to prevent timeouts
        return self.has_voted_locally(voter_name)

    def record_vote(self, voter_name: str, candidate_id: str, tx_hash: str = None):
        with open("votes.txt", "a") as file:
            tx_info = f"tx: {tx_hash}" if tx_hash else "blockchain_failed"
            file.write(f"{voter_name}: {candidate_id} ({tx_info})\n")

        self.voted_users.add(voter_name)
        self._save_voted_users()

    def get_stats(self) -> Dict[str, int]:
        total_registered = len(self.face_registry)
        total_voted = len(self.voted_users)
        return {
            "total_registered": total_registered,
            "total_voted": total_voted,
            "remaining_voters": total_registered - total_voted
        }

# Global instance
voter_service = VoterService()
