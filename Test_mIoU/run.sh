model=4000
while (($model<21000))
do
    echo $model
    /usr/local/MATLAB/R2016b/bin/matlab -nodisplay -nosplash -r  "argument='$model';ValLabelEvalSegResults"
    model=`expr $model + 4000`
done
