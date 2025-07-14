import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description='power forecast')
    parser.add_argument('train_file', help='training data file path')
    parser.add_argument('test_file', help='testing data file path')
    parser.add_argument('--target', help='target prediction column name', default='Global_active_power')
    parser.add_argument('--features', help='list of features for prediction, comma separated', default='')
    parser.add_argument('--short', action='store_true', help='short-term prediction (90 days)')
    parser.add_argument('--long', action='store_true', help='long-term prediction (365 days)')
    parser.add_argument('--experiments', type=int, help='number of experiment rounds', default=5)
    parser.add_argument('--multiscale', action='store_true', help='plot prediction comparison')
    parser.add_argument('--use-gpu', action='store_true', help='use GPU acceleration')
    parser.add_argument('--use-cpu', action='store_true', help='force using CPU')
    parser.add_argument('--save-model', action='store_true', help='save trained model')
    parser.add_argument('--multi-feature', action='store_true', help='use multi-feature prediction')
    parser.add_argument('--auto-select', action='store_true', help='automatically select best features')
    parser.add_argument('-m', '--model', type=str, choices=['lstm', 'transformer'], help='choose model', default='lstm')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.train_file):
        print(f"Error: Training file '{args.train_file}' does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.test_file):
        print(f"Error: Testing file '{args.test_file}' does not exist")
        sys.exit(1)
    
    try:
        print("Reading training and testing data...")
        train_data = pd.read_csv(args.train_file)
        test_data = pd.read_csv(args.test_file)
        
        # Ensure date column is in datetime format
        train_data['date'] = pd.to_datetime(train_data['date'])
        test_data['date'] = pd.to_datetime(test_data['date'])
        
        print(f"Training data size: {len(train_data)} days")
        print(f"Testing data size: {len(test_data)} days")
        print(f"Target column: {args.target}")
        
        predictor = Predictor(sequence_length=90)
        
        predictor.set_device(use_gpu=args.use_gpu, use_cpu=args.use_cpu)
        
        feature_columns = None
        if args.features:
            feature_columns = [f.strip() for f in args.features.split(',')]
            print(f"Using specified features: {', '.join(feature_columns)}")
        elif args.multi_feature:
            feature_columns = [col for col in train_data.columns 
                              if col not in ['date', args.target]]
            print(f"Using all available features: {', '.join(feature_columns)}")
        
        if args.auto_select and (args.multi_feature or feature_columns):
            print("Performing automatic feature selection (based on training data)...")
            corr_matrix = train_data.corr()[args.target].abs()
           
            corr_matrix = corr_matrix.drop(args.target).sort_values(ascending=False)
            
            selected_features = corr_matrix[corr_matrix > 0.1].index.tolist()
            
            print(f"Automatically selected features: {', '.join(selected_features)}")
            feature_columns = selected_features
        
        run_short = args.short or (not args.short and not args.long)
        run_long = args.long or (not args.short and not args.long)
        run_multiscale = args.multiscale
        #use_multi_feature = args.multi_feature or feature_columns is not None
        if run_short:
            result = predictor.run_multifeature_prediction_with_plot(
                train_data, test_data, args.target, feature_columns, 
                prediction_days=90, prediction_type='short', model_name=args.model
                , multiscale=run_multiscale
            )
            print(f"Short-term prediction MSE: {result['mse']:.4f}, MAE: {result['mae']:.4f}")
        if args.save_model:
            predictor.save_model('short')
        
        if run_long:
            result = predictor.run_multifeature_prediction_with_plot(
                train_data, test_data, args.target, feature_columns, 
                prediction_days=365, prediction_type='long', model_name=args.model
                , multiscale=run_multiscale
            )
            print(f"Long-term prediction MSE: {result['mse']:.4f}, MAE: {result['mae']:.4f}")

        
    except Exception as e:
        print(f"Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage: python lstm_predictor.py <training_file> <testing_file> [options]")
        print("Short-term prediction: python lstm_predictor.py daily_train.csv daily_test.csv --short")
        print("Long-term prediction: python lstm_predictor.py daily_train.csv daily_test.csv --long")
        print("Multi-feature prediction: python lstm_predictor.py daily_train.csv daily_test.csv --multi-feature")
        print("Specify features: python lstm_predictor.py daily_train.csv daily_test.csv --features 'feature1,feature2'")
        print("Auto-select features: python lstm_predictor.py daily_train.csv daily_test.csv --multi-feature --auto-select")
        print("Plot comparison: python lstm_predictor.py daily_train.csv daily_test.csv --plot")
        print("Multiple experiments: python lstm_predictor.py daily_train.csv daily_test.csv --experiments 10")
        print("Use GPU: python lstm_predictor.py daily_train.csv daily_test.csv --use-gpu")
        print("Save model: python lstm_predictor.py daily_train.csv daily_test.csv --save-model")
