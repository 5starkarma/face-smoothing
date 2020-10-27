import unittest
import shutil
import tempfile
import mock

from utils.video import delete_file


class TestVideoFunctions(unittest.TestCase):
 
    @mock.patch('utils.video.os')
    def test_delete_file(self, mock_os):
        delete_file('any path')
        # test that rm called os.remove with the right parameters
        mock_os.remove.assert_called_with('any path')

    def test_make_temp_dir(self):
        pass

    def test_split_video(self):
        pass

    def test_process_video(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
