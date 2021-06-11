import os
import shutil


class AssetManager:

	def __init__(self, base_dir):
		self.__base_dir = base_dir

		self.__cache_dir = os.path.join(self.__base_dir, 'cache')
		if not os.path.exists(self.__cache_dir):
			os.mkdir(self.__cache_dir)

		self.__preprocess_dir = os.path.join(self.__cache_dir, 'preprocess')
		if not os.path.exists(self.__preprocess_dir):
			os.mkdir(self.__preprocess_dir)

		self.__models_dir = os.path.join(self.__cache_dir, 'models')
		if not os.path.exists(self.__models_dir):
			os.mkdir(self.__models_dir)

	def get_preprocess_file_path(self, data_name):
		return os.path.join(self.__preprocess_dir, data_name + '.npz')

	def get_model_dir(self, model_name):
		return os.path.join(self.__models_dir, model_name)

	def recreate_model_dir(self, model_name):
		model_dir = self.get_model_dir(model_name)

		self.__recreate_dir(model_dir)
		return model_dir

	@staticmethod
	def __recreate_dir(path):
		if os.path.exists(path):
			shutil.rmtree(path)

		os.makedirs(path)
