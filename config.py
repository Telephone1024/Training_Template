import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # Data settings
    parser.add_argument('--batch_size', type=int,
                        default=8,
                        help='batch size on a single GPU')
    parser.add_argument('--data_path', type=str,
                        default='',
                        help='Path to dataset')
    parser.add_argument('--img_size', type=int,
                        default=224,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int,
                        default=16,
                        help='Number of data loading threads')
    
    # Model settings
    parser.add_argument('--resume', type=str,
                        # default='',
                        help='Checkpoint to resume')
    
    # Training settings
    parser.add_argument('--train_stage', type=str, choices=['train', 'finetune'],
                        default='train',
                        help='Training stage')
    parser.add_argument('--num_epochs', type=int,
                        default=50)
    parser.add_argument('--num_warmup_epochs', type=int,
                        default=0)
    parser.add_argument('--weight_decay', type=float,
                        default=0.05)
    parser.add_argument('--lr', type=float,
                        default=5e-4)
    parser.add_argument('--warmup_lr', type=float,
                        default=5e-7)
    parser.add_argument('--min_lr', type=float,
                        default=5e-6)
    parser.add_argument('--clip_grad', type=float,
                        default=5.0)
    
    # Scheduler settings
    parser.add_argument('--sche_type', type=str, choices=['cosine', 'step'],
                        default='cosine',
                        help='Learning rate scheduler for training')
    # # For step scheduler
    parser.add_argument('--decay_epochs', type=int,
                        default=2,
                        help='Epoch interval to decay LR, used in StepLRScheduler')
    parser.add_argument('--decay_rate', type=float,
                        default=0.9,
                        help='LR decay rate, used in StepLRScheduler')
    
    # Optimizer settings
    parser.add_argument('--optim', type=str, choices=['sgd', 'adamw'],
                        default='adamw')
    parser.add_argument('--eps', type=float,
                        default=1e-8)
    parser.add_argument('--betas', type=tuple,
                        default=(0.9, 0.999))
    parser.add_argument('--momentum', type=float,
                        default=0.9,
                        help='SGD momentum')
    
    # Other settings
    parser.add_argument('--saved_path', type=str,
                        default='./logs/',
                        help='Path to save checkpoints and logs')
    parser.add_argument('--val_interval', type=int,
                        default=1,
                        help='Validate interval epoch number')
    parser.add_argument('--save_interval', type=int,
                        default=10,
                        help='Interval between saving checkpoints')
    parser.add_argument('--print_interval', type=int,
                        default=50,
                        help='Interval between logging')
    
    args = parser.parse_args()
    return args
