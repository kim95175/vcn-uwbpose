

## Human Detection - train


#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr4_512 --hidden_dim 512 --device cuda:1
#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr3_supermix --num_queries 20 --device cuda:1 --mixup_prob 0.7 --model_debug


#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr5_basem--device cuda:1

#python main.py --batch_size 64 --epochs 50 --pose hm --output_dir weights/rftr0_mlphm/ --frozen_weights weights/rftr0/rftr0_200.pth --model_debug



#python main.py --batch_size 64 --epochs 50 --pose hmdr --output_dir weights/rftr0_mlpdr_roi/ --frozen_weights weights/base200.pth --feature 16 --roi --model_debug

#python main.py --batch_size 128 --epochs 50 --pose simdr --output_dir weights/rftr0_mlpsimdr/ --frozen_weights weights/rftr0/rftr0_200.pth --device cuda:1 --model_debug


#python main.py --batch_size 128 --epochs 50 --pose hm --output_dir weights/rftr0_mlp/ --frozen_weights weights/rftr0/rftr0_200.pth --model_debug


#python main.py --epochs 200 --batch_size 96 --output_dir weights/vcq2 --feature 16 --model_debug
#python main.py --epochs 200 --batch_size 96 --output_dir weights/vcq2_mlp --box_feature 16 --lr 0.0004 --device cuda:1

#python main.py --epochs 200 --batch_size 96 --output_dir weights/vcq2_mlp88 --box_feature 16 
#python main.py --epochs 200 --batch_size 96 --output_dir weights/vcq2_mlp88_nomix --box_feature 16 --mixup_prob 0. --device cuda:1 --model_debug

#python main.py --epochs 200 --batch_size 96 --output_dir weights/vcq2_mlpsrc --box_feature 16 --resume weights/vcq2_mlpsrc/checkpoint0009.pth --device cuda:1 --model_debug 


#python main.py --batch_size 64 --pose hmdr --epochs 50 --frozen_weights weights/rftr15_q15/checkpoint0199.pth --output_dir weights/rftr15_posehmdr_ssim/ --num_queries 15 --feature 16 --model_debug
#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr2_44 --num_txrx 4 --device cuda:1

#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr0 --model_debug


#Test   
#for var in 66 67 68 #69 
#for var in 71 72 76 77
for var in 65 70 
#for var in 52 53 #55 65 70 #65
#for var in 51 52 53 54 55 61 62 63 64 65 66 67 68 69 70
do
    #python main.py --batch_size 32 --eval --output_dir weights/rftr0/ --resume weights/rftr0/rftr0_200.pth --test_dir $var
    #python main.py --batch_size 32 --eval --vis --output_dir weights/rftr0/ --resume weights/rftr0/rftr0_200.pth --img_dir results/rftr0/nms_x/$var/  --test_dir $var
    

    #python main.py --batch_size 32 --eval --vis --output_dir weights/rftr0/ --resume weights/rftr0/rftr0_200.pth --img_dir results/rftr0/nms_x50/$var/ --test_dir $var --box_threshold 0.8
    #python main.py --batch_size 32 --eval --vis --output_dir weights/rftr0/ --resume weights/rftr0/rftr0_200.pth --img_dir results/rftr0/nms_o50/$var/ --test_dir $var --soft_nms
    #python main.py --batch_size 32 --eval --vis --output_dir weights/rftr0/ --resume weights/rftr0/rftr0_200.pth --img_dir results/rftr0/nms_o80/$var/ --test_dir $var --soft_nms --box_threshold 0.8
    #python main.py --batch_size 32 --eval --vis --output_dir weights/vcq2/ --resume weights/vcq2/checkpoint0039.pth --img_dir results/0708/vcq2_nmsO_80_$var/ --test_dir $var --device cuda:1 --feature 16 --soft_nms --box_threshold 0.8
    #python main.py --batch_size 64 --eval --output_dir weights/vcq2/ --resume weights/vcq2/checkpoint0039.pth --test_dir $var --device cuda:1 --feature 16 --soft_nms --box_threshold 0.8
    
    #python main.py --batch_size 32 --eval --vis --output_dir weights/rftr3_supermix/ --resume weights/rftr3_supermix/checkpoint0039.pth --img_dir results/0708/rftr3_$var/ --test_dir $var --device cuda:1
    python main.py --batch_size 64 --eval --output_dir weights/rftr3_supermix/ --resume weights/rftr3_supermix/checkpoint0189.pth --test_dir $var --device cuda:1 --num_queries 20
    
    #python main.py --batch_size 64 --pose hmdr --eval --vis --output_dir weights/rftr0_mlpdr/ --resume weights/rftr0_mlpdr/checkpoint0039.pth --test_dir $var --img_dir results/0710/mlphmdr/$var/ --soft_nms

    # w/ feature
    #python main.py --batch_size 64 --pose hm --eval --vis --output_dir weights/rftr0_mlphm/ --resume weights/rftr0_mlphm/checkpoint0049.pth --test_dir $var --img_dir results/rftr0/mlphm/$var/ --box_threshold 0.8
    #python main.py --batch_size 64 --pose hm --eval --vis --output_dir weights/rftr0_mlphmftr/ --resume weights/rftr0_mlphmftr/checkpoint0019.pth --test_dir $var --feature 16 --img_dir results/rftr0/mlphmftr/$var/ --box_threshold 0.8
    
    #python main.py --batch_size 64 --pose hmdr --eval --vis --output_dir weights/rftr0_mlpdr/ --resume weights/rftr0_mlpdr/checkpoint0039.pth --test_dir $var --img_dir results/rftr0_posehmdr/nms_x/ --device cuda:1
    #python main.py --batch_size 64 --pose hmdr --eval --vis --output_dir weights/rftr0_mlpdr/ --resume weights/rftr0_mlpdr/checkpoint0039.pth --test_dir $var --img_dir results/rftr0_posehmdr/nms_o/ --soft_nms --device cuda:1
    
    #python main.py --batch_size 128 --pose hmdr --eval --vis --output_dir weights/vcq2_hmdr/ --resume weights/vcq2_hmdr/checkpoint0039.pth --test_dir $var --img_dir results/0710/vcq2_hmdr_$var --box_feature 16 --soft_nms 
    
done
