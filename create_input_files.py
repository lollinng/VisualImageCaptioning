from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flicker30k',
                       karpathy_json_path='data_sets/dataset_flickr30k.json',
                       image_folder='data_sets/flickr30k_images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder="data_sets/caption_data/",
                       max_len=50)