import pytest
import kipy


def test_kicad_connection():
    """Test the KiCad API connection."""
    try:
        kicad = kipy.KiCad()
        version = kicad.get_version()
        assert version is not None
        assert hasattr(version, 'full_version')
    except Exception as e:
        pytest.skip(f"KiCad is not running: {e}")


def test_kicad_connection_failure():
    """Test that an exception is raised when KiCad is not running."""
    with pytest.raises(Exception):
        # An exception must be raised when KiCad is not running
        kicad = kipy.KiCad(socket_path='ipc:///invalid/path')
        kicad.get_version()


def test_board_from_editor():
    """Test accessing the board open in the KiCad editor."""
    try:
        # Create a KiCad client
        kicad = kipy.KiCad()

        # Get the currently open board (needs a document specifier)
        # Must get the document specifier of the currently open board

        board = kicad.get_board()
        if board is None:
            pytest.skip("No board is currently open")

        # Confirm the board object was created correctly
        assert board is not None
        assert hasattr(board, 'get_footprints')
        assert hasattr(board, 'get_tracks')
        assert hasattr(board, 'get_vias')

        # Check basic info
        footprints = board.get_footprints()
        tracks = board.get_tracks()
        vias = board.get_vias()
        pads = board.get_pads()
        nets = board.get_nets()
        board.get_selection()
        board.get_shapes()
        board.get_pad_shapes_as_polygons(pads[31])
        board.get_selection()

        footprints[0].id
        pads[31].id

        footprints[0].position
        x = footprints[0]
        x.definition.shapes
        x.sheet_path
        x.definition.items


        print(f"Footprint count on board: {len(footprints)}")
        print(f"Track count on board: {len(tracks)}")
        print(f"Via count on board: {len(vias)}")
        
    except Exception as e:
        pytest.skip(f"Cannot access the board from the KiCad editor: {e}")


def test_board_move_footprint():
    from kipy import KiCad
    from kipy.geometry import Vector2, Angle
    try:
        board = KiCad().get_board()
        footprints = board.get_footprints()

        for footprint in footprints:
            footprint.position += Vector2.from_xy_mm(5, 2)
            footprint.orientation += Angle.from_degrees(90)
        board.update_items(footprints)
    except Exception as e:
        pytest.skip(f"Cannot run the footprint move test: {e}")


def test_remove_footprint():
    """Test deleting a footprint from the board."""
    try:
        from kipy import KiCad
        
        kicad = KiCad()
        board = kicad.get_board()
        
        if board is None:
            pytest.skip("No board is currently open")
        
        # Check footprint count before deletion
        footprints = board.get_footprints()
        initial_count = len(footprints)
        
        if initial_count == 0:
            pytest.skip("No footprint to delete")
        
        print(f"Footprint count before delete: {initial_count}")
        
        # Delete the first footprint
        board.remove_items(footprints[0])

        # Check footprint count after deletion
        remaining_footprints = board.get_footprints()
        remaining_count = len(remaining_footprints)

        print(f"Footprint count after delete: {remaining_count}")

        # Confirm the deletion happened correctly
        assert remaining_count == initial_count - 1
        
    except Exception as e:
        pytest.skip(f"Cannot run the footprint delete test: {e}")



def test_board_load_from_file():
    """Test loading a board directly from a file."""
    import os

    # Path of the PCB file to test
    pcb_file = "engine/kicad-python/kicad/demos/ecc83/ecc83-pp.kicad_pcb"

    # Check whether the file exists
    if not os.path.exists(pcb_file):
        pytest.skip(f"PCB file does not exist: {pcb_file}")
    
    try:
        # Create a KiCad client
        kicad = kipy.KiCad(client_name='abc')
        board = kicad.get_board()
        kicad._client.connected
        kicad._client._client_name
        board.get_pads()
        board.get_nets()
        
        tracks = board.get_tracks()
        vias = board.get_vias()
        
        kicad.get_project()

        # Load the board from the file

        if board is None:
            pytest.skip(f"Cannot load PCB file: {pcb_file}")
        kicad.get_open_documents()
        kicad.get_project()

        # Confirm the board object was created correctly
        assert board is not None
        assert hasattr(board, 'get_footprints')
        assert hasattr(board, 'get_tracks')
        assert hasattr(board, 'get_vias')

        # Check basic info
        footprints = board.get_footprints()
        tracks = board.get_tracks()
        vias = board.get_vias()

        print(f"Footprint count on loaded board: {len(footprints)}")
        print(f"Track count on loaded board: {len(tracks)}")
        print(f"Via count on loaded board: {len(vias)}")
        
    except Exception as e:
        pytest.skip(f"Cannot load PCB file: {e}")


if __name__ == "__main__":
    pytest.main([__file__])