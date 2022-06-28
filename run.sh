

## Human Detection - train

#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr15_q15 --device cuda:1
#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr0_dec9 --dec_layers 9 --resume weights/rftr0_dec9/checkpoint0129.pth

#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr2_44 --num_txrx 4 --device cuda:1

#python main.py --epochs 200 --batch_size 128 --output_dir weights/rftr0 --model_debug

#python main.py --batch_size 96 --pose simdr --epochs 50 --frozen_weights weights/rftr15_q15/checkpoint0199.pth --output_dir weights/rftr15_pose200/
#python main.py --batch_size 96 --pose simdr --epochs 50 --frozen_weights weights/rftr15_q15/checkpoint0199.pth --output_dir weights/rftr15_poseftr200/ --feature 16

#python main.py --batch_size 64 --pose hmdr --epochs 50 --frozen_weights weights/rftr15_q15/checkpoint0199.pth --output_dir weights/rftr15_posehmdr/ --device cuda:1

#Test   
#for var in 66 67 68 #69 
#for var in 71 72 76 77
#for var in 55 65 70
for var in 65
do
    python main.py --batch_size 32 --eval --output_dir weights/rftr0/ --resume weights/rftr0/rftr0_200.pth --test_dir $var

    # w/ feature
    #python main.py --batch_size 64 --pose simdr --eval --output_dir weights/rftr15_poseftr200/ --resume weights/rftr15_poseftr200/checkpoint0049.pth --test_dir $var --feature 16
    #python main.py --batch_size 64 --pose simdr --eval --vis --output_dir weights/rftr15_poseftr200/ --resume weights/rftr15_poseftr200/checkpoint0049.pth --test_dir $var --feature 16 --img_dir results/rftr15_feature16/$var/ --box_threshold 0.8
    
    #python main.py --batch_size 64 --pose simdr --eval --output_dir weights/rftr15_pose200/ --resume weights/rftr15_pose200/checkpoint0049.pth --test_dir $var
    #python main.py --batch_size 16 --pose simdr --eval --vis --output_dir weights/rftr15_pose200/ --resume weights/rftr15_pose200/checkpoint0049.pth --test_dir $var --img_dir results/rftr15_pose/$var/ --box_threshold 0.8
    
    #python main.py --batch_size 64 --pose hm --eval --vis --output_dir weights/rftr15_posehm/ --resume weights/rftr15_posehm/checkpoint0049.pth --test_dir $var --img_dir results/rftr15/posehm/$var/ --box_threshold 0.8
    #python main.py --batch_size 64 --pose hm --eval --vis --output_dir weights/rftr15_posehmftr/ --resume weights/rftr15_posehmftr/checkpoint0039.pth --test_dir $var --feature 16 --img_dir results/rftr15/posehmftr/$var/ --box_threshold 0.8
    
    #python main.py --batch_size 64 --pose hmdr --eval --output_dir weights/rftr15_posehmdr/ --resume weights/rftr15_posehmdr/checkpoint0049.pth --test_dir $var
    #python main.py --batch_size 64 --pose hmdr --eval --output_dir weights/rftr15_posehmdr_ftr/ --resume weights/rftr15_posehmdr_ftr/checkpoint0049.pth --test_dir $var --feature 16 
    
done

