import os
import shutil
from tqdm import tqdm

######################################################
path = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_class_attack_adv"  #1063194
classes_path = os.listdir(path)

for cla in tqdm(classes_path[0:100]):
    imgs_path = os.path.join(path,cla)
    imgs = os.listdir(imgs_path)
    for img in imgs:
        new_folder = os.path.join(path,cla,img)
        imgs_12 = os.listdir(new_folder)
        for image in imgs_12:
            # if "_attack_1_" in image:
            #     # print(image)

            #     target_image_path =  os.path.join(path,cla,img,image)

            #     cla_sor = cla
            #     class_len = len(cla)
            #     cla_tar = image.split("_attack_1_")[0][class_len+1:]

            #     source_image_name = cla_tar+"_"+img.split("_")[-1]+".JPEG"
            #     # if cla=="tiger":
            #     #     source_image_name = cla+"_"+cla_tar+"_"+img.split("_")[-1]+".JPEG"
            #     #     print(source_image_name)
            #     target_image_name = image
            #     source_image_path = os.path.join(path,cla,img,source_image_name)
            #     target_image_path = os.path.join(path,cla,img,image)

            #     new_name = cla+"__"+cla_tar+"__"+new_folder.split("_")[-1]+".JPEG"
            #     shutil.copy(source_image_path,"/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/source/"+new_name)
            #     shutil.copy(target_image_path,"/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/target/"+new_name)

            #     # try:
            #     #     shutil.copy(source_image_path,"/data/sunchangchang/MSU/PGD_attack/ImageNet_final/train_controlnet/source/"+new_name)
            #     #     shutil.copy(target_image_path,"/data/sunchangchang/MSU/PGD_attack/ImageNet_final/train_controlnet/target/"+new_name)
            #     # except OSError:
            #     #     print(img,image,"1")
            #     #     print(source_image_path,"2")
            #     #     print(target_image_path,"2")
            #             # shutil.copy(os.path.join(path,cla,img,cla+"_"+cla_tar+"_"+img.split("_")[-1]+".JPEG"),os.path.join(path,cla,img,source_image_name))
            #     # print(source_image_path,"1")
            #     # print(target_image_path,"2")
            #     # # print(image,"1")
            #     # print(new_name,"3")

            #     break
            #######################################
            # if "_attack_1_" in image:
            #     # print(image)

            #     target_image_path =  os.path.join(path,cla,img,image)

            #     cla_sor = cla
            #     class_len = len(cla)
            #     cla_tar = image.split("_attack_1_")[0][class_len+1:]

            #     source_image_name = cla_tar+"_"+img.split("_")[-1]+".JPEG"
            #     # if cla=="tiger":
            #     #     source_image_name = cla+"_"+cla_tar+"_"+img.split("_")[-1]+".JPEG"
            #     #     print(source_image_name)
            #     target_image_name = image
            #     source_image_path = os.path.join(path,cla,img,source_image_name)
            #     target_image_path = os.path.join(path,cla,img,image)
                 
            #     new_name = cla+"__"+cla_tar+"__"+new_folder.split("_")[-1]+".JPEG"
            #     # shutil.copy(source_image_path,"/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/source2/"+new_name)
            #     shutil.copy(target_image_path,"/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/benign1/"+new_name)
            #     # shutil.copy(target_image_path2,"/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/benign2/"+new_name)

            #     # try:
            #     #     shutil.copy(source_image_path,"/data/sunchangchang/MSU/PGD_attack/ImageNet_final/train_controlnet/source/"+new_name)
            #     #     shutil.copy(target_image_path,"/data/sunchangchang/MSU/PGD_attack/ImageNet_final/train_controlnet/target/"+new_name)
            #     # except OSError:
            #     #     print(img,image,"1")
            #     #     print(source_image_path,"2")
            #     #     print(target_image_path,"2")
            #             # shutil.copy(os.path.join(path,cla,img,cla+"_"+cla_tar+"_"+img.split("_")[-1]+".JPEG"),os.path.join(path,cla,img,source_image_name))
            #     # print(source_image_path,"1")
            #     # print(target_image_path,"2")
            #     # # print(image,"1")
            #     # print(new_name,"3")

            #     # break
            # ############
            # if "_attack_2_" in image:
            #     # print(image)

            #     target_image_path =  os.path.join(path,cla,img,image)

            #     cla_sor = cla
            #     class_len = len(cla)
            #     cla_tar = image.split("_attack_2_")[0][class_len+1:]

            #     source_image_name = cla_tar+"_"+img.split("_")[-1]+".JPEG"
            #     # if cla=="tiger":
            #     #     source_image_name = cla+"_"+cla_tar+"_"+img.split("_")[-1]+".JPEG"
            #     #     print(source_image_name)
            #     target_image_name = image
            #     source_image_path = os.path.join(path,cla,img,source_image_name)
            #     target_image_path = os.path.join(path,cla,img,image)
                
            #     new_name = cla+"__"+cla_tar+"__"+new_folder.split("_")[-1]+".JPEG"
            #     # shutil.copy(source_image_path,"/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/source2/"+new_name)
            #     # shutil.copy(target_image_path,"/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/benign1/"+new_name)
            #     shutil.copy(target_image_path,"/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/benign2/"+new_name)

            #     # try:
            #     #     shutil.copy(source_image_path,"/data/sunchangchang/MSU/PGD_attack/ImageNet_final/train_controlnet/source/"+new_name)
            #     #     shutil.copy(target_image_path,"/data/sunchangchang/MSU/PGD_attack/ImageNet_final/train_controlnet/target/"+new_name)
            #     # except OSError:
            #     #     print(img,image,"1")
            #     #     print(source_image_path,"2")
            #     #     print(target_image_path,"2")
            #             # shutil.copy(os.path.join(path,cla,img,cla+"_"+cla_tar+"_"+img.split("_")[-1]+".JPEG"),os.path.join(path,cla,img,source_image_name))
            #     # print(source_image_path,"1")
            #     # print(target_image_path,"2")
            #     # # print(image,"1")
            #     # print(new_name,"3")

            #     # break

###############################################

            if "_attack_1_" in image:
                # print(image)

                # target_image_path =  os.path.join(path,cla,img,image)

                cla_sor = cla
                class_len = len(cla)
                cla_tar = image.split("_attack_1_")[0][class_len+1:]

                source_image_name = cla_tar+"_"+img.split("_")[-1]+".JPEG"
                # if cla=="tiger":
                #     source_image_name = cla+"_"+cla_tar+"_"+img.split("_")[-1]+".JPEG"
                #     print(source_image_name)
                target_image_name = image
                source_image_path = os.path.join(path,cla,img,source_image_name)

                target_image_path = os.path.join(path,cla,img,img+".JPEG")

                new_name = cla+"__"+cla_tar+"__"+new_folder.split("_")[-1]+".JPEG"
                shutil.copy(source_image_path,"/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/3source_adv/"+new_name)
                shutil.copy(target_image_path,"/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/3source_original/"+new_name)

                # try:
                #     shutil.copy(source_image_path,"/data/sunchangchang/MSU/PGD_attack/ImageNet_final/train_controlnet/source/"+new_name)
                #     shutil.copy(target_image_path,"/data/sunchangchang/MSU/PGD_attack/ImageNet_final/train_controlnet/target/"+new_name)
                # except OSError:
                #     print(img,image,"1")
                #     print(source_image_path,"2")
                #     print(target_image_path,"2")
                        # shutil.copy(os.path.join(path,cla,img,cla+"_"+cla_tar+"_"+img.split("_")[-1]+".JPEG"),os.path.join(path,cla,img,source_image_name))
                # print(source_image_path,"1")
                # print(target_image_path,"2")
                # # print(image,"1")
                # print(new_name,"3")

                break




######################################################
# import json
# # # ##############iamgenet
# # prompt_cifar10 = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/prompt_imagenet_noprompt_new.json"
# prompt_cifar10 = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/prompt_imagenet_prompt_new.json"

# path = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/source"
# path_t = "/egr/research-optml/sunchan5/AEG/ImageNet/PGD_attack/ImageNet_final/train_controlnet/target"
# images = os.listdir(path)
# with open(prompt_cifar10, "w") as json_lines_file:
#     for img in images:
#     #dog__cat__44538.png
#         # print(img)
#         img_dict = {}
#         img_dict["source"] = os.path.join(path,img)
#         img_dict["target"] = os.path.join(path_t,img)
#         label_s = img.split("__")[0]
#         label_t = img.split("__")[1]
#         # img_dict["prompt"] = "An image of "+label_s+" is classified as "+label_t
#         img_dict["prompt"] = "An image of "+label_s
#         # img_dict["prompt"] = " "
#         img_dict["ground_truth"] = label_s+"\t"+label_t
#         json.dump(img_dict, json_lines_file)
#         json_lines_file.write("\n")
