"""Role-based access control enforcement for ResisTrack."""

from __future__ import annotations

from resistrack.auth.models import ROLE_PERMISSIONS, UserRole, UserSession


class AccessDeniedError(Exception):
    """Raised when a user lacks the required permission."""

    def __init__(self, user_id: str, permission: str, role: UserRole) -> None:
        self.user_id = user_id
        self.permission = permission
        self.role = role
        super().__init__(
            f"Access denied: user={user_id} role={role} "
            f"lacks permission={permission}"
        )


class RBACEnforcer:
    """Enforce role-based access control."""

    @staticmethod
    def check_permission(session: UserSession, permission: str) -> bool:
        """Check if a user session has the required permission."""
        return permission in session.permissions

    @staticmethod
    def require_permission(session: UserSession, permission: str) -> None:
        """Require a permission, raising AccessDeniedError if absent."""
        if permission not in session.permissions:
            raise AccessDeniedError(
                user_id=session.user_id,
                permission=permission,
                role=session.role,
            )

    @staticmethod
    def get_role_permissions(role: UserRole) -> frozenset[str]:
        """Get all permissions for a given role."""
        return ROLE_PERMISSIONS.get(role, frozenset())

    @staticmethod
    def has_any_permission(session: UserSession, permissions: set[str]) -> bool:
        """Check if session has any of the specified permissions."""
        return bool(session.permissions & permissions)

    @staticmethod
    def has_all_permissions(session: UserSession, permissions: set[str]) -> bool:
        """Check if session has all of the specified permissions."""
        return permissions.issubset(session.permissions)


__all__ = ["RBACEnforcer", "AccessDeniedError"]
