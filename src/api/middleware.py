"""
SENTINEL-AI — API Middleware
JWT Bearer auth, body size limit, process time header.
"""

import os
import time
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# JWT via python-jose
try:
    from jose import JWTError, jwt
except ImportError:
    jwt = None
    JWTError = Exception

SECRET_KEY = os.environ.get("JWT_SECRET", "sentinel-dev-secret-change-in-prod")
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
MAX_BODY_SIZE = 5000  # characters

security = HTTPBearer(auto_error=False)


async def verify_jwt(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[dict]:
    """
    Verify JWT token from Authorization: Bearer header.

    In development mode (JWT_SECRET=sentinel-dev-secret-change-in-prod),
    auth is optional. In production, valid JWT is required.
    """
    is_dev = SECRET_KEY == "sentinel-dev-secret-change-in-prod"

    if credentials is None:
        if is_dev:
            return {"sub": "dev-user", "role": "admin"}
        raise HTTPException(status_code=401, detail="Missing authorization header")

    token = credentials.credentials

    if jwt is None:
        if is_dev:
            return {"sub": "dev-user", "role": "admin"}
        raise HTTPException(status_code=500, detail="python-jose not installed")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """Add X-Process-Time header to all responses."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies exceeding MAX_BODY_SIZE characters."""

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            if len(body) > MAX_BODY_SIZE * 4:  # rough char-to-byte ratio
                return Response(
                    content='{"detail":"Request body too large"}',
                    status_code=413,
                    media_type="application/json",
                )
        response = await call_next(request)
        return response
