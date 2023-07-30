import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from tqdm import tqdm
from PIL import Image
import relational_image_generation_evaluation as rige

WORKING_DIR = '../struct_IF/results/CC500'

evaluator = rige.Evaluator('ViT-L/14', device='cuda:0')
#evaluator = rige.Evaluator('ViT-L/14-Datacomp', device='cuda:0')
adv_dataset = rige.get_cc500_graph_dataloader().dataset
score_tag = 'attr_scores'
#orig_prompts, adv_prompts = rige.get_adv_prompt_list('attributes')

orig_images = []
orig_correct_graphs = []
orig_wrong_graphs = []

dirs = []
for fname in os.listdir(WORKING_DIR):
    if os.path.isdir(os.path.join(WORKING_DIR,fname)):
        dirs.append(os.path.join(WORKING_DIR,fname))
if len(dirs) == 0:
    dirs = [WORKING_DIR]

for IMAGE_DIR in dirs:
    img_names = os.listdir(IMAGE_DIR)
    img_names.sort()
    for image_name in tqdm(img_names[:6], desc='loading data...'):
        if not image_name.endswith('_I.png'):
            continue
        ident, seed = image_name.split('_')[0], image_name.split('_')[1]
        ident_id = int(ident)
        if ident_id > 431:
            continue
        for obj in adv_dataset[ident_id].labels.values():
            if obj not in rige.FILTERED_OBJECTS:
                continue

        if ident_id % 2 == 0:
            adv_ident_id = ident_id + 1
        else:
            adv_ident_id = ident_id - 1

        temp = Image.open(os.path.join(IMAGE_DIR, image_name))
        img = temp.copy()
        temp.close()
        orig_images.append(img)
        orig_correct_graphs.append(adv_dataset[ident_id])
        orig_wrong_graphs.append(adv_dataset[adv_ident_id])

    print('calculating scores...')
    og_correct_scores = evaluator(orig_images, orig_correct_graphs)
    og_wrong_scores = evaluator(orig_images, orig_wrong_graphs)

    acc = 0
    for correct_score, wrong_score in zip(og_correct_scores[score_tag], og_wrong_scores[score_tag]):
        if correct_score > wrong_score:
            acc += 1
    og_acc = acc/len(og_correct_scores[score_tag])


    print('\n--- {} ---------------'.format(os.path.basename(IMAGE_DIR)))
    print('og accuracy: \t{:.1f}'.format(og_acc*100))
