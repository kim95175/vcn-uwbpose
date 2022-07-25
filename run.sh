

## Human Detection - train

#python main.py --epochs 200 --batch_size 96 --output_dir weights/base_l --model l --device cuda:1 --resume weights/base_l/checkpoint0039.pth --model_debug 
#python main.py --epochs 300 --batch_size 128 --output_dir weights/base --device cuda:1 --model_debug 



#python main.py --batch_size 64 --epochs 50 --pose hmdr --output_dir weights/rftr0_hmdr_ftrgram/ --frozen_weights weights/base200.pth --feature 16 --device cuda:1 --model_debug
#python main.py --batch_size 64 --epochs 50 --pose hmdr --output_dir weights/rftr0_hmdr_ftr2/ --frozen_weights weights/base200.pth --feature 16 --model_debug


#python main.py --batch_size 96 --epochs 50 --pose hmdr --output_dir weights/rftr0_hmdr_mlpftr/ --frozen_weights weights/base200.pth --feature 16 --model_debug
#python main.py --batch_size 80 --epochs 50 --pose hmdr --output_dir weights/rftr0_hmdr_mlpftr32/ --frozen_weights weights/base200.pth --feature 32 --model_debug
#python main.py --batch_size 80 --epochs 50 --pose hmdr --output_dir weights/rftr0_hmdr_onlyftr32/ --frozen_weights weights/base200.pth --feature 32 --device cuda:1 --model_debug
#python main.py --batch_size 96 --epochs 50 --pose hmdr --output_dir weights/rftr0_hmdr_onlyftr16/ --frozen_weights weights/base200.pth --feature 16 --model_debug




python main.py --batch_size 128 --epochs 200 --pose hmdr --output_dir weights/rftr0_ftr_train/ --feature 16 --feature_train --device cuda:1 --resume weights/rftr0_ftr_train/checkpoint_ftr0099.pth
#python main.py --batch_size 128 --epochs 50 --pose hmdr --output_dir weights/rftr0_hmdr_freezeftr/ --frozen_weights weights/base200.pth --frozen_ftr_weights weights/rftr0_ftr_train/checkpoint_ftr0099.pth --model_debug




#python main.py --batch_size 64 --pose hmdr --epochs 50 --frozen_weights weights/rftr15_q15/checkpoint0199.pth --output_dir weights/rftr15_posehmdr_ssim/ --num_queries 15 --feature 16 --model_debug
#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr2_44 --num_txrx 4 --device cuda:1

#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr0 --model_debug


#Test   
#for var in 66 67 68 #69 
#for var in 71 72 76 77
#for var in 42 #55 65 70 
for var in 44 45 46 47 48 49 50 51
#for var in 51 52 53 54 55 61 62 63 64 65 66 67 68 69 70
do
    #python main.py --batch_size 16 --eval --output_dir weights/rftr0/ --resume weights/rftr0/rftr0_200.pth --test_dir $var
    #python main.py --batch_size 32 --eval --vis --output_dir weights/rftr0/ --resume weights/rftr0/rftr0_200.pth --img_dir results/rftr0/nms_x/$var/  --test_dir $var
    
    #python main.py --batch_size 16 --eval --output_dir weights/vcq2_mlpsrc/ --resume weights/vcq2_mlpsrc/checkpoint0149.pth --box_feature 16 --test_dir $var
    #python main.py --batch_size 16 --eval --output_dir weights/vcq2_mlpsrc/ --resume weights/vcq2_mlpsrc/checkpoint0149.pth --box_feature 16 --test_dir $var --soft_nms
    #python main.py --batch_size 32 --eval --vis --output_dir weights/vcq2_mlpsrc/ --resume weights/vcq2_mlpsrc/checkpoint0149.pth --img_dir results/rftr0/nms_x/$var/ --test_dir $var
    

    #python main.py --batch_size 64 --eval --output_dir weights/base200/ --resume weights/base200.pth --test_dir $var
    


    # w/ feature
    #python main.py --batch_size 64 --pose hm --eval --vis --output_dir weights/rftr0_mlphm/ --resume weights/rftr0_mlphm/checkpoint0049.pth --test_dir $var --img_dir results/rftr0/mlphm/$var/ --box_threshold 0.8
    #python main.py --batch_size 64 --pose hm --eval --vis --output_dir weights/rftr0_mlphmftr/ --resume weights/rftr0_mlphmftr/checkpoint0019.pth --test_dir $var --feature 16 --img_dir results/rftr0/mlphmftr/$var/ --box_threshold 0.8
    
    #python main.py --batch_size 64 --pose hmdr --eval --vis --output_dir weights/rftr0_hmdr/ --resume weights/rftr0_hmdr/checkpoint0029.pth --test_dir $var --img_dir results/0720/base_hmdr_$var
    #python main.py --batch_size 32 --pose hmdr --eval --vis --output_dir weights/rftr0_hmdr/ --resume weights/rftr0_hmdr/checkpoint0029.pth --test_dir $var --img_dir results/0721/xy_test_$var/ --box_threshold 0.8
    
    #python main.py --batch_size 64 --pose hmdr --eval --vis --output_dir weights/rftr0_hmdr_ftr/ --resume weights/rftr0_hmdr_ftr/checkpoint0029.pth --test_dir $var --feature 16  --img_dir results/0718/base_hdmr_ftr_$var
    
    #python main.py --batch_size 128 --pose hmdr --eval --vis --output_dir weights/vcq2_hmdr/ --resume weights/vcq2_hmdr/checkpoint0039.pth --test_dir $var --img_dir results/0710/vcq2_hmdr_$var --box_feature 16 --soft_nms 
    #python main.py --batch_size 128 --pose hmdr --eval --vis --output_dir weights/vcq_src_hmdr/ --resume weights/vcq_src_hmdr/checkpoint0029.pth --test_dir $var --img_dir results/0717/vcqsrc_hmdr_$var --box_feature 16 #--soft_nms 
    #python main.py --batch_size 128 --pose hmdr --eval --vis --output_dir weights/vcq_src_hmdr/ --resume weights/vcq_src_hmdr/checkpoint0029.pth --test_dir $var --img_dir results/0717/vcqsrc_hmdr_nms_$var --box_feature 16 --soft_nms 
    
done
