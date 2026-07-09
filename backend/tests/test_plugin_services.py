"""Unit tests for the plugin repository and service layers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.exceptions import ConflictError, ResourceNotFoundError
from app.database.repositories.plugin_repository import PluginRepository
from app.domain.plugin import PluginSpec, PluginStatus
from app.services.plugin_service import PluginService

# =========================================================================
# PluginRepository
# =========================================================================


class TestPluginRepository:
    @pytest.fixture
    def session(self) -> AsyncMock:
        s = AsyncMock()
        s.add = MagicMock()
        return s

    @pytest.fixture
    def repo(self, session: AsyncMock) -> PluginRepository:
        return PluginRepository(session)

    # -- find_by_name ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_find_by_name_found(self, repo: PluginRepository, session: AsyncMock) -> None:
        from app.database.models.plugin import PluginModel

        model = PluginModel(
            id=str(uuid4()),
            name="test-plugin",
            display_name="Test Plugin",
            version="1.0.0",
            sdk_version=">=0.1.0",
            description="A test plugin",
            author="tester",
            trust_tier="community",
            enabled=True,
            install_type="entry_point",
            status="active",
            status_message="",
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = model
        session.execute.return_value = mock_result  # sync return from async session.execute()

        spec = await repo.find_by_name("test-plugin")
        assert spec is not None
        assert spec.name == "test-plugin"
        assert spec.display_name == "Test Plugin"
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_find_by_name_not_found(self, repo: PluginRepository, session: AsyncMock) -> None:
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        spec = await repo.find_by_name("missing")
        assert spec is None

    # -- list_enabled ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_enabled(self, repo: PluginRepository, session: AsyncMock) -> None:
        from app.database.models.plugin import PluginModel

        m1 = PluginModel(id=str(uuid4()), name="p1", enabled=True,
                          display_name="P1", version="1.0.0", sdk_version=">=0.1.0",
                          description="", author="", trust_tier="community",
                          install_type="entry_point", status="active", status_message="")
        m2 = PluginModel(id=str(uuid4()), name="p2", enabled=True,
                          display_name="P2", version="1.0.0", sdk_version=">=0.1.0",
                          description="", author="", trust_tier="community",
                          install_type="entry_point", status="active", status_message="")

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [m1, m2]

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        session.execute.return_value = mock_result

        specs = await repo.list_enabled()
        assert len(specs) == 2
        assert specs[0].name == "p1"
        assert specs[1].name == "p2"

    @pytest.mark.asyncio
    async def test_list_enabled_empty(self, repo: PluginRepository, session: AsyncMock) -> None:
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []

        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        session.execute.return_value = mock_result

        specs = await repo.list_enabled()
        assert specs == []

    # -- update_status -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_update_status_found(self, repo: PluginRepository, session: AsyncMock) -> None:
        plugin_id = str(uuid4())
        from app.database.models.plugin import PluginModel

        mock_execute = MagicMock()
        mock_execute.rowcount = 1
        session.execute.return_value = mock_execute

        model = PluginModel(id=plugin_id, name="p1", status="active",
                            display_name="", version="0.1.0", sdk_version=">=0.1.0",
                            description="", author="", trust_tier="community",
                            enabled=False, install_type="entry_point",
                            status_message="")
        session.get = AsyncMock(return_value=model)

        spec = await repo.update_status(plugin_id, "active")
        assert spec is not None
        assert spec.status == "active"

    @pytest.mark.asyncio
    async def test_update_status_not_found(self, repo: PluginRepository, session: AsyncMock) -> None:
        mock_execute = MagicMock()
        mock_execute.rowcount = 0
        session.execute.return_value = mock_execute

        spec = await repo.update_status("missing", "active")
        assert spec is None

    # -- upsert (insert) ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_upsert_insert(self, repo: PluginRepository, session: AsyncMock) -> None:
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        spec = PluginSpec(name="new-plugin")
        result = await repo.upsert(spec)
        assert result.name == "new-plugin"
        assert session.add.called

    # -- upsert (update) ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_upsert_update(self, repo: PluginRepository, session: AsyncMock) -> None:
        from app.database.models.plugin import PluginModel

        existing_id = str(uuid4())
        existing_model = PluginModel(
            id=existing_id,
            name="existing",
            display_name="Old Name",
            version="0.1.0",
            sdk_version=">=0.1.0",
            description="",
            author="",
            trust_tier="community",
            enabled=False,
            install_type="entry_point",
            status="installed",
            status_message="",
        )

        mock_find_result = MagicMock()
        mock_find_result.scalar_one_or_none.return_value = existing_model
        session.execute.return_value = mock_find_result

        spec = PluginSpec(name="existing", display_name="New Name")

        # The BaseRepository.update() does: merged = await self._session.merge(model)
        # We need merge to return a properly populated PluginModel.
        from app.database.models.plugin import PluginModel
        session.merge = AsyncMock(return_value=PluginModel(
            id=existing_id,
            name="existing",
            display_name="New Name",
            version="0.1.0",
            sdk_version=">=0.1.0",
            description="",
            author="",
            trust_tier="community",
            enabled=False,
            install_type="entry_point",
            status="installed",
            status_message="",
        ))

        result = await repo.upsert(spec)
        assert result.name == "existing"
        assert result.display_name == "New Name"


# =========================================================================
# PluginService
# =========================================================================


class TestPluginService:
    @pytest.fixture
    def repo(self) -> AsyncMock:
        return AsyncMock(spec=PluginRepository)

    @pytest.fixture
    def service(self, repo: AsyncMock) -> PluginService:
        return PluginService(repository=repo)

    # -- install -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_install_creates_plugin(
        self, service: PluginService, repo: AsyncMock,
    ) -> None:
        repo.find_by_name = AsyncMock(return_value=None)
        repo.add = AsyncMock(side_effect=lambda spec: spec)

        result = await service.install(
            name="my-plugin",
            display_name="My Plugin",
            version="1.0.0",
        )

        assert result.name == "my-plugin"
        assert result.display_name == "My Plugin"
        assert result.version == "1.0.0"
        repo.add.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_install_duplicate_name_raises(
        self, service: PluginService, repo: AsyncMock,
    ) -> None:
        existing = PluginSpec(name="dup")
        repo.find_by_name = AsyncMock(return_value=existing)

        with pytest.raises(ConflictError, match="already exists"):
            await service.install(name="dup")

    # -- uninstall ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_uninstall_deletes(self, service: PluginService, repo: AsyncMock) -> None:
        repo.delete = AsyncMock(return_value=True)

        await service.uninstall("some-id")
        repo.delete.assert_awaited_once_with("some-id")

    @pytest.mark.asyncio
    async def test_uninstall_not_found_raises(
        self, service: PluginService, repo: AsyncMock,
    ) -> None:
        repo.delete = AsyncMock(return_value=False)

        with pytest.raises(ResourceNotFoundError, match="not found"):
            await service.uninstall("missing")

    # -- get ---------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_found(self, service: PluginService, repo: AsyncMock) -> None:
        expected = PluginSpec(id="p1", name="test")
        repo.get = AsyncMock(return_value=expected)

        result = await service.get("p1")
        assert result.id == "p1"
        assert result.name == "test"

    @pytest.mark.asyncio
    async def test_get_not_found_raises(
        self, service: PluginService, repo: AsyncMock,
    ) -> None:
        repo.get = AsyncMock(return_value=None)

        with pytest.raises(ResourceNotFoundError, match="not found"):
            await service.get("missing")

    # -- get_by_name -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_by_name_found(self, service: PluginService, repo: AsyncMock) -> None:
        expected = PluginSpec(name="test-plugin")
        repo.find_by_name = AsyncMock(return_value=expected)

        result = await service.get_by_name("test-plugin")
        assert result.name == "test-plugin"

    @pytest.mark.asyncio
    async def test_get_by_name_not_found_raises(
        self, service: PluginService, repo: AsyncMock,
    ) -> None:
        repo.find_by_name = AsyncMock(return_value=None)

        with pytest.raises(ResourceNotFoundError, match="not found"):
            await service.get_by_name("missing")

    # -- list_installed ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_installed(self, service: PluginService, repo: AsyncMock) -> None:
        specs = [PluginSpec(name="a"), PluginSpec(name="b")]
        repo.list_all = AsyncMock(return_value=specs)

        result = await service.list_installed()
        assert len(result) == 2

    # -- list_enabled ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_enabled(self, service: PluginService, repo: AsyncMock) -> None:
        specs = [PluginSpec(name="a", enabled=True)]
        repo.list_enabled = AsyncMock(return_value=specs)

        result = await service.list_enabled()
        assert len(result) == 1
        assert result[0].enabled is True

    # -- update ------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_update_fields(self, service: PluginService, repo: AsyncMock) -> None:
        original = PluginSpec(id="p1", name="test", display_name="Old")
        repo.get = AsyncMock(return_value=original)
        repo.update = AsyncMock(side_effect=lambda spec: spec)

        updated = await service.update("p1", display_name="New Name")
        assert updated.display_name == "New Name"
        repo.update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_not_found_raises(
        self, service: PluginService, repo: AsyncMock,
    ) -> None:
        repo.get = AsyncMock(return_value=None)

        with pytest.raises(ResourceNotFoundError, match="not found"):
            await service.update("missing", display_name="New")

    # -- activate / deactivate ---------------------------------------------

    @pytest.mark.asyncio
    async def test_activate(self, service: PluginService, repo: AsyncMock) -> None:
        original = PluginSpec(id="p1", name="test", enabled=False, status="installed")
        repo.get = AsyncMock(return_value=original)
        repo.update = AsyncMock(side_effect=lambda spec: spec)

        result = await service.activate("p1")
        assert result.enabled is True
        assert result.status == PluginStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_deactivate(self, service: PluginService, repo: AsyncMock) -> None:
        original = PluginSpec(id="p1", name="test", enabled=True, status="active")
        repo.get = AsyncMock(return_value=original)
        repo.update = AsyncMock(side_effect=lambda spec: spec)

        result = await service.deactivate("p1")
        assert result.enabled is False
        assert result.status == PluginStatus.DISABLED
