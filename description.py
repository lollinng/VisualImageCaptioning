import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize
import imageio
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def caption_image_beam_search(encoder,decoder,image_path,word_map,beam_size=3):
    """
    Reads an image and caption it with beam search

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param bream_size: number of sequences to consider at each decode_step
    :return caption,weights for visualisation
    """

    k = beam_size
    vocab_size = len(word_map)

    # REad image and process 
    img = imageio.imread(image_path)
    if len(img.shape)==2:
        img = img[:,:,np.newaxis]
        img = np.concatenate([img,img,img],axis=2)


    # img = imresize(img,(256,256))
    img = img.transpose(2,0,1)
    img = img/255
    
    img = torch.FloatTensor(img).to(device)
    if(img.size()[0]!=3):
        print('sorry the model doesn\'t support images apart from rbg data pls try another image')
        return None
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3,256,256)

    # Encode
    image = image.unsqueeze(0)      # (1,3,256,256) # for batch size
    encoder_out = encoder(image)    # (1,enc_image,enc_image_size,encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # flatten encoding
    encoder_out = encoder_out.view(1,-1,encoder_dim)  # (1,num_pixels,encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k,num_pixels,encoder_dim) # (k,num_pixels,encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  #(k,1)

    # Tensor to store top k sequnces; now they're just <start>
    seqs = k_prev_words  # (k,1)

    # Tensor to store top k sequnces' scores; now they're just 0
    top_k_scores = torch.zeros(k,1).to(device) # (k,1)

    # Lists to store completed sequences. thier alphas and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s,embed_dim)
        
        awe,alpha = decoder.attention(encoder_out,h) # (s,encoder_dim),(s,num_pixels)

        alpha = alpha.view(-1,enc_image_size,enc_image_size) # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h)) # getting scalar, (s,encoder_dim)
        awe = gate*awe

        h,c = decoder.decode_step(torch.cat([embeddings,awe],dim=1),(h,c)) #(s,decoder_dim)

        scores = decoder.fc(h) # (s,vocab_size)
        scores = F.log_softmax(scores,dim=1)

        # Add 
        scores = top_k_scores.expand_as(scores) + scores # (s,vocab_size)

        # for the first step, all k points will have the same scores (since smake k previous words,h,c)
        if step==1:
            top_k_scores,top_k_words = scores[0].topk(k,0,True,True) #s
        else:
            # unroll and find top scores, and thier unrolled indices
            top_k_scores,top_k_words = scores.view(-1).topk(k,0,True,True) #(s)
        
        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words//vocab_size # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new word to sequence,aphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)



         # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    return seq



if __name__ == '__main__':

    
    model = 'BEST_checkpoint_flicker30k_5_cap_per_img_5_min_word_freq.pth.tar'
    word_map = 'data_sets/caption_data/WORDMAP_flicker30k_5_cap_per_img_5_min_word_freq.json'

    beam_size = 5
    path = "test"
    questions = os.listdir(path)
    answers = []
    checkpoint = torch.load(model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    for img in questions:
        # Load model
        # img = 'test/dog.jpg'
        img = 'test/'+ img
        

        # Load word map (word2ix)
        with open(word_map, 'r') as j:
            word_map_dict = json.load(j)
        rev_word_map = {v: k for k, v in word_map_dict.items()}  # ix2word
        # Encode, decode with attention and beam search
        seq= caption_image_beam_search(encoder, decoder, img, word_map_dict, beam_size)
        print([rev_word_map[ind] for ind in seq])
    # for i i 
 
