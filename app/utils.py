import hashlib
import json

def hash_features(features_dict: dict) -> str:
    """Crée un hash unique pour les features afin de gérer le cache"""
    return hashlib.md5(
        json.dumps(features_dict, sort_keys=True).encode()
    ).hexdigest()