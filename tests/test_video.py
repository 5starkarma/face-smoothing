import unittest
import shutil
import tempfile
import mock

from utils.video import (delete_file,
                         split_video)


class VideoTestCase(unittest.TestCase):
 
    @mock.patch('utils.video.os.path')
    @mock.patch('utils.video.os')
    def test_delete_file(self, mock_os, mock_path):
        # Set up the mock
        mock_path.isfile.return_value = False
        # Delete file
        delete_file('any path')
        # Test that the delete call was not called.
        self.assertFalse(mock_os.remove.called, "Failed to not remove the file if not present.")
        # Make the file 'exist'
        mock_path.isfile.return_value = True
        # Delete file
        delete_file('any path')
        # Test that the delete call was called.
        mock_os.remove.assert_called_with('any path')

    def test_split_video(self, test_video_file):
        # Take input filename and output list of np.arrays
        pass

    def test_process_video(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
