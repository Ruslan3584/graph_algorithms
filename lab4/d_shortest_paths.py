import argparse
import os, sys
import time, json, string
import numpy as np
import cv2
from numba import njit


def string_to_image(string,reference_images,noise_level):
    # create string as array
    image = reference_images[string[0]]
    for i in string[1:]:
        image = np.hstack([image,reference_images[i]])
    n,m = image.shape

    # generate binomial noise
    ksi = np.random.binomial(size=n*m, n=1, p=noise_level).reshape(image.shape)
    output_image = ksi^image
    return output_image


def import_images(folder_path,alphabet_list):
    reference_images = {}
    for i in alphabet_list[:-1] + ['space']:
        img = (cv2.imread(folder_path + f'/{i}.png',cv2.IMREAD_GRAYSCALE)/255).astype(int)
        reference_images[i] = img
    reference_images[' '] = reference_images.pop('space')
    return reference_images


def string_to_image(string,reference_images,noise_level):
    # create string as array
    image = reference_images[string[0]]
    for i in string[1:]:
        image = np.hstack([image,reference_images[i]])
    n,m = image.shape
    # generate binomial noise
    ksi = np.random.binomial(size=n*m, n=1, p=noise_level).reshape(image.shape)
    output_image = ksi^image
    return output_image     



def get_bigrams(json_path):
    with open(json_path) as json_file: 
        frequencies_dict = json.load(json_file)

    # create alphabet
    alphabet_list = list(string.ascii_lowercase + ' ')
    alphabet_dict = dict((j,i) for i,j in enumerate(alphabet_list))

    # all pairs as array
    array = np.zeros([len(alphabet_list),len(alphabet_list)]).astype('int')
    for i in alphabet_dict:
        for j in alphabet_dict:
            if i + j in frequencies_dict:
                array[alphabet_dict[i]][alphabet_dict[j]] = frequencies_dict[i+j]

    # make a-priopi probabilities from frequencies
    p_k = (array.T/array.sum(axis=1)).T

    # make -np.inf where p(k) = 0
    p_k = - np.log(p_k, out=np.full_like(-p_k, -np.inf), where=(p_k!=0))

    return (alphabet_list, alphabet_dict, p_k)



def get_unary_penalties(img_string, alphabet_list, reference_images, p):

    number_of_letters = int(img_string.shape[1]/img_string.shape[0])
    letters_probab = np.zeros((number_of_letters, len(alphabet_list)))

    height = img_string.shape[0]
    for i, start_ind_letter in enumerate(range(0, img_string.shape[1], height)):
        xored_letters = img_string[:,start_ind_letter:start_ind_letter+height]^list(reference_images.values())
        letters_probab[i,...] =  -np.sum((xored_letters)*np.log(p) + (1^xored_letters)*np.log(1-p), axis = (1,2))

    return letters_probab


def get_best_d_paths(pointers_d, alphabet_arr):
    best = []
    d = pointers_d.shape[2]
    for path in range(0,d):
        best_last = pointers_d[-1,0,path]
        result = np.empty(pointers_d.shape[0], dtype=int)
        result[-1] = best_last
        for j in range(pointers_d.shape[0]-2, -1, -1):
            if j == 0:
                for best_last in range(d):
                    result[j] = pointers_d[j,result[j+1]][best_last]
                    if list(result) not in best:
                        best.append(list(result))
                        break
            result[j] = pointers_d[j,result[j+1]][0]
    
    out_strings = [''.join(alphabet_arr[x]) for x in best]
    return out_strings

def get_d_shortest(d, letters_probab, alphabet_arr, p_k):
    number_of_letters = letters_probab.shape[0]
    f = np.zeros((number_of_letters,len(alphabet_arr),d ))
    f[0,:,0] = p_k[-1,:]+ letters_probab[0, :]
    f[0,:,1:] = np.inf
    pointers_d = np.full((number_of_letters, len(alphabet_arr), d), -1).astype(int)
    for letter in range(number_of_letters - 1):
        temp = []
        for next_label in range(len(alphabet_arr)):
            all_paths = np.zeros((len(alphabet_arr),d))
            for prev_label in range(len(alphabet_arr)):
                all_paths[prev_label,:] = f[letter,prev_label,:] + p_k[prev_label, next_label] + letters_probab[letter + 1, next_label]
            
            temp.append(all_paths)
            f[letter + 1,next_label,:] = np.sort(all_paths.ravel())[:d]
            pointers_d[letter,next_label,:] = np.argsort(np.min(all_paths,axis=1))[:d]
    pointers_d[-1,:,:] = (np.argsort(f[-1,...].ravel())//d)[:d]

    d_shortest = get_best_d_paths(pointers_d, alphabet_arr)

    return d_shortest


def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("input_string", type=str, help="input string")
    parser.add_argument("noise_level", type=float, help="noise level of bernoulli distribution")
    parser.add_argument("d", type=int, help="number of shortest paths")
    args = parser.parse_args()

    alphabet_list, alphabet_dict, p_k = get_bigrams('frequencies.json')
    reference_images = import_images('alphabet',alphabet_list)
    alphabet_arr = np.array(alphabet_list)


    noised_image = string_to_image(args.input_string,reference_images,args.noise_level)

    letters_probab = get_unary_penalties(noised_image, alphabet_list, reference_images, args.noise_level)

    d_best_strings = get_d_shortest(args.d, letters_probab, alphabet_arr, p_k)

    print(d_best_strings)


if __name__ == "__main__":
    main()
