import argparse
import os
from tqdm import tqdm
from PIL import Image
import relational_image_generation_evaluation as rige


def run(args):
    model = 'ViT-L/14'
    evaluator = rige.Evaluator(model, device='cuda:0')
    score_tag = 'attr_scores'

    # the images, graphs and adversarial graphs are loaded
    images = []
    correct_graphs = []
    adv_graphs = []
    img_names = os.listdir(args.image_dir)
    img_names.sort()
    if args.dataset == 'CC':
        adv_dataset = rige.get_cc500_graph_dataloader().dataset

        for image_name in tqdm(img_names, desc='loading data...'):
            if not (image_name.endswith('.png') or image_name.endswith('.jpg')):
                continue
            ident, seed = image_name.split('_')[0], image_name.split('_')[1]
            ident_id = int(ident)
            if ident_id > 431:
                continue

            wrong_obj = False
            for obj in adv_dataset[ident_id].labels.values():
                if obj not in rige.FILTERED_OBJECTS:
                    wrong_obj = True
                    break
            if wrong_obj:
                continue

            if ident_id % 2 == 0:
                adv_ident_id = ident_id + 1
            else:
                adv_ident_id = ident_id - 1

            temp = Image.open(os.path.join(args.image_dir, image_name))
            img = temp.copy()
            temp.close()
            images.append(img)
            correct_graphs.append(adv_dataset[ident_id])
            adv_graphs.append(adv_dataset[adv_ident_id])

    elif args.dataset == 'DAA':
        adv_dataset = rige.get_adversarial_attribute_dataset()
        for image_name in tqdm(img_names, desc='loading data...'):
            if not (image_name.endswith('.png') or image_name.endswith('.jpg')):
                continue
            ident, seed = image_name.split('_')[0], image_name.split('_')[1]
            ident_id, ident_ds = ident.split('-')
            ident_id = int(ident_id)
            if ident_ds == 'og':
                temp = Image.open(os.path.join(args.image_dir, image_name))
                img = temp.copy()
                temp.close()
                images.append(img)
                correct_graphs.append(adv_dataset[ident_id]['original_graph'])
                adv_graphs.append(adv_dataset[ident_id]['adv_graph'])
            elif ident_ds == 'adv':
                temp = Image.open(os.path.join(args.image_dir, image_name))
                img = temp.copy()
                temp.close()
                images.append(img)
                correct_graphs.append(adv_dataset[ident_id]['adv_graph'])
                adv_graphs.append(adv_dataset[ident_id]['original_graph'])
            else:
                assert False

    # calculating the similarity scores and EPViT accuracy
    print('nb of found images: {}'.format(len(images)))
    print('calculating scores...')
    og_correct_scores = evaluator(images, correct_graphs)
    og_adv_scores = evaluator(images, adv_graphs)

    acc = 0
    for correct_score, adv_score in zip(og_correct_scores[score_tag], og_adv_scores[score_tag]):
        if correct_score > adv_score:
            acc += 1
    epvit_acc = acc / len(og_correct_scores[score_tag])
    print('\n-----------------------------------------------------')
    print('EPViT accuracy: \t{:.1f}'.format(epvit_acc * 100))
    print('-----------------------------------------------------\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['DAA', 'CC'], required=True)
    parser.add_argument("--image-dir", type=str, required=True)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
