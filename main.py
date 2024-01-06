from dataPreprocess import preprocess_data , transform_text
from linear_regression import LinearRegression
from logistic_reression import LogisticRegression
from mnb import MultinomialNaiveBayes
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


def accuracy_score(y_pred,y_test):
    count=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_test[i]:
            count+=1
    return count/len(y_pred)


''' I have imported the classes from the respective files and used them in the main function below to train and test the models 
if you want to run the code for a particular model just comment out the other models and run the code for the model you want to run.
The results and graphs of every classifier are present in my report '''


if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test,vectorizer = preprocess_data()
    lin_reg=LinearRegression()
    log_reg=LogisticRegression()
    mnb = MultinomialNaiveBayes()
    
    
    
    #-----------------------------------------------------------------------------------------------------------
    # Linear regression
    # epochs=1000
    # learning_rate=0.01
    # w, b, train_errors, valid_errors, test_mse= lin_reg.train_linear_regression(X_train, y_train, X_valid, y_valid, X_test, y_test, learning_rate, epochs)
    # _,acc_test_lin=lin_reg.predict(X_test,y_test,w,b)
    # _,acc_valid_lin=lin_reg.predict(X_valid,y_valid,w,b)
    # best_lr, best_accuracy,precision_best,recall_best,f1_best,accuracies,lrs,precisions,recalls,f1s,epochs_values_list, best_epoch_value=lin_reg.hyperparameter_tuning_Lin(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # print("Detailed results for linear regression")
    # print("Test Accuracy using linear regression",acc_test_lin)
    # print("Validation Accuracy using linear regression",acc_valid_lin)
    # print("Test MSE using linear regression",test_mse)
    
    # print("-----------------------------------------------------------------------------------------------------------")
    # plt.figure(figsize=(8, 5))
    # epochsg = range(1, len(train_errors) + 1)
    # plt.plot(epochsg, train_errors, marker='o', label='Training Error')
    # plt.plot(epochsg, valid_errors, marker='o', label='Validation Error')

    # plt.title('Training and Validation Errors Over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Error')
    # plt.legend()
    # # Save the figure
    # plt.tight_layout()
    # plt.savefig('training_validation_errors.png')
    # plt.show()
    
    
    # print("-----------------------------------------------------------------------------------------------------------")    
    # print("After hyperparameter tuning")
    # print("Best Epoch Value:", best_epoch_value)
    # print("Best Learning Rate:", best_lr)
    # print("Best Accuracy:", best_accuracy)
    # print("Precision:", precision_best)
    # print("Recall:", recall_best)
    # print("F1 score:", f1_best)
    # for i in range(len(lrs)):
    #     print("for Learning Rate:",lrs[i],"Epochs:",epochs_values_list[i],"Accuracy:",accuracies[i],"Precision:",precisions[i],"Recall:",recalls[i],"F1 score:",f1s[i])
    # print("-----------------------------------------------------------------------------------------------------------")
    # print("length of lists",len(lrs),len(epochs_values_list),len(accuracies),len(precisions),len(recalls),len(f1s))    
    # #-----------------------------------------------------------------------------------------------------------
    # plt.figure(figsize=(10, 6))

    # plt.plot(epochs_values_list, accuracies, marker='o', label='Accuracy')
    # plt.plot(epochs_values_list, lrs, marker='o', label='Learning Rate')
    # plt.plot(epochs_values_list, precisions, marker='o', label='Precision')
    # plt.plot(epochs_values_list, recalls, marker='o', label='Recall')
    # plt.plot(epochs_values_list, f1s, marker='o', label='F1 Score')

    # plt.title('Performance Metrics Over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Metric Value')
    # plt.legend()

    # # Save the figure
    # plt.tight_layout()
    # plt.savefig('linear_regression_combined_graph.png')
    # plt.show()
    # plt.savefig('linear_regression_graphs.png')
    #-----------------------------------------------------------------------------------------------------------
    

    #-----------------------------------------------------------------------------------------------------------
    # Logistic regression #Done
    # epochs_log=50
    # learning_rate=0.01
    # w, b, train_accuracies, valid_accuracy, test_accuracy= log_reg.train_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test, learning_rate, epochs_log)
    # best_learning_rate,best_epoch_value,epochs_values_list, best_accuracy,accuracies,lrs,precision_best,precisions,recall_best,recalls,f1_best,f1s=log_reg.hyperparameter_tuning(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # print("Detailed results for logistic regression")
    # print("Test Accuracy using logistic regression",test_accuracy)
    
    
    # plt.figure(figsize=(8, 5))
    # epochs1 = range(1, len(train_accuracies) + 1)
    # plt.plot(epochs1, train_accuracies, marker='o', label='Training Accuracy')
    # plt.plot(epochs1, valid_accuracy, marker='o', label='Validation Accuracy')

    # plt.title('Training and Validation Accuracies Over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # # Save the figure
    # plt.tight_layout()
    # plt.savefig('training_validation_accuracies.png')
    # plt.show()
    # print("After hyperparameter tuning")
    # print("Best Learning Rate:", best_learning_rate)
    # print("Best Epoch Value:", best_epoch_value)
    # print("Best Accuracy:", best_accuracy)
    # print("Precision:", precision_best)
    # print("Recall:", recall_best)
    # print("F1 score:", f1_best)

    # data = []
    # for i in range(len(lrs)):
    #     row = [lrs[i], epochs_values_list[i], accuracies[i], precisions[i], recalls[i], f1s[i]]
    #     data.append(row)

    # headers = ["Learning Rate", "Epochs", "Accuracy", "Precision", "Recall", "F1 Score"]

    # table = PrettyTable(headers)

    # for row in data:
    #     table.add_row(row)

    # print(table)

# -----------------------------------------------------------------------------------------------------------
    # plt.figure(figsize=(10, 6))

    # plt.plot(epochs_values_list, accuracies, marker='o', label='Accuracy')
    # plt.plot(epochs_values_list, lrs, marker='o', label='Learning Rate')
    # plt.plot(epochs_values_list, precisions, marker='o', label='Precision')
    # plt.plot(epochs_values_list, recalls, marker='o', label='Recall')
    # plt.plot(epochs_values_list, f1s, marker='o', label='F1 Score')

    # plt.title('Performance Metrics Over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Metric Value')
    # plt.legend()

    # # Save the figure
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('logistic_regression_graphs.png')
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------  
# Multinomial naive bayes
    # Fit the model with the best alpha
    # mnb.fit(X_train, y_train, alpha=0.5)

    # # Perform predictions on the test set
    # y_pred_test = mnb.predict(X_test)
    # acc_mnb=accuracy_score(y_pred_test,y_test)
    # # Hyperparameter tuning
    # best_alpha, best_accuracy,accuracies_list,alpha_list,precision_best,precision_list,recall_best,recall_list,f1_best,f1s_list = mnb.hyperparameter_tuning(X_train, y_train, X_valid, y_valid)
    # print("Test Accuracy using multinomial naive bayes",acc_mnb)
    # print("After hyperparameter tuning")
    # print("Best alpha:", best_alpha)
    # print("Best accuracy:", best_accuracy)
    # print("Precision:", precision_best)
    # print("Recall:", recall_best)
    # print("F1 score:", f1_best)
    # for i in range(len(alpha_list)):
    #     print("For Alpha:",alpha_list[i],"Accuracy:",accuracies_list[i],"Precision:",precision_list[i],"Recall:",recall_list[i],"F1 score:",f1s_list[i])
    # plt.figure(figsize=(8, 5))
    
# -----------------------------------------------------------------------------------------------------------
    # # Save the figure
    # plt.figure(figsize=(10, 6))

    # plt.plot(alpha_list, accuracies_list, marker='o', label='Accuracy')
    # plt.plot(alpha_list, precision_list, marker='o', label='Precision')
    # plt.plot(alpha_list, recall_list, marker='o', label='Recall')
    # plt.plot(alpha_list, f1s_list, marker='o', label='F1 Score')

    # plt.xscale('log')  # Use log scale for better visualization if alpha values vary widely
    # plt.title('Evaluation Metrics vs. Alpha')
    # plt.xlabel('Alpha')
    # plt.ylabel('Metric Value')
    # plt.legend()
    # plt.grid(True)

    # plt.show()
# -----------------------------------------------------------------------------------------------------------

    ''' I am using mnb with best alpha value that i got from tuning for prediction as it has the highest accuracy
    To check if my model is working or not i am taking input from the user and predicting if it is spam or not
    just pass any spam or ham message and it will predict if it is spam or not
    for you convenience I have provided some spam messages below
    '''
    
    '''SPAM MESSAGES: Congratulations! You've won a free iPhone. Click the link to claim your prize now! 
                URGENT: You have been selected for a special offer! Click now to claim your exclusive prize!
'''
    mnb.fit(X_train, y_train, alpha=0.1)
    x=mnb.predict(X_test)
    acc=accuracy_score(x,y_test)
    print("Test Accuracy using multinomial naive bayes",acc)

    input_sms = input("Enter the message")

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = vectorizer.transform([transformed_sms])
    dense_vector_input = vector_input.toarray()
    # # 3. predict
    result = mnb.predict(dense_vector_input)[0]
    # 4. Display
    if result == 1:
        print("Spam")
    else:
        print("Not Spam")
            




    
    
    


    