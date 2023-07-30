import os
from tqdm import tqdm
from PIL import Image
import relational_image_generation_evaluation as rige

WORKING_DIR = '../struct_IF/results/old/DAA20-if'
eval_relationships = False

evaluator = rige.Evaluator('ViT-L/14', device='cuda:0')
#evaluator = rige.Evaluator('ViT-L/14-Datacomp', device='cuda:0')
if eval_relationships:
    adv_dataset = rige.get_adversarial_relationship_dataset()
    score_tag = 'rel_scores'
else:
    adv_dataset = rige.get_adversarial_attribute_dataset()
    score_tag = 'attr_scores'
    #orig_prompts, adv_prompts = rige.get_adv_prompt_list('attributes')

orig_images = []
orig_correct_graphs = []
orig_wrong_graphs = []
adv_images = []
adv_correct_graphs = []
adv_wrong_graphs = []

dirs = []
for fname in os.listdir(WORKING_DIR):
    if os.path.isdir(os.path.join(WORKING_DIR,fname)):
        dirs.append(os.path.join(WORKING_DIR,fname))
if len(dirs) == 0:
    dirs = [WORKING_DIR]

for IMAGE_DIR in dirs:
    img_names = os.listdir(IMAGE_DIR)
    img_names.sort()
    for image_name in tqdm(img_names, desc='loading data...'):
        if not image_name.endswith('_I.png'):
            continue
        ident, seed = image_name.split('_')[0], image_name.split('_')[1]
        ident_id, ident_ds = ident.split('-')
        ident_id = int(ident_id)
        if ident_ds == 'og':
            temp = Image.open(os.path.join(IMAGE_DIR, image_name))
            img = temp.copy()
            temp.close()
            orig_images.append(img)
            orig_correct_graphs.append(adv_dataset[ident_id]['original_graph'])
            orig_wrong_graphs.append(adv_dataset[ident_id]['adv_graph'])
        elif ident_ds == 'adv':
            temp = Image.open(os.path.join(IMAGE_DIR, image_name))
            img = temp.copy()
            temp.close()
            adv_images.append(img)
            adv_correct_graphs.append(adv_dataset[ident_id]['adv_graph'])
            adv_wrong_graphs.append(adv_dataset[ident_id]['original_graph'])
        else:
            assert False

    print('calculating scores...')
    og_correct_scores = evaluator(orig_images, orig_correct_graphs)
    og_wrong_scores = evaluator(orig_images, orig_wrong_graphs)
    adv_correct_scores = evaluator(adv_images, adv_correct_graphs)
    adv_wrong_scores = evaluator(adv_images, adv_wrong_graphs)

    acc = 0
    for correct_score, wrong_score in zip(og_correct_scores[score_tag], og_wrong_scores[score_tag]):
        if correct_score > wrong_score:
            acc += 1
    og_acc = acc/len(og_correct_scores[score_tag])

    acc = 0
    for correct_score, wrong_score in zip(adv_correct_scores[score_tag], adv_wrong_scores[score_tag]):
        if correct_score > wrong_score:
            acc += 1
    adv_acc = acc/len(adv_correct_scores[score_tag])

    print('\n--- {} ---------------'.format(os.path.basename(IMAGE_DIR)))
    print('og accuracy: \t{:.1f}'.format(og_acc*100))
    print('adv accuracy: \t{:.1f}'.format(adv_acc*100))
    print('avg accuracy: \t{:.1f}'.format((adv_acc + og_acc)*50))
    print('overleaf: \t & {:.1f} & {:.1f} & {:.1f}\n'.format(og_acc*100, adv_acc*100, (adv_acc + og_acc)*50))