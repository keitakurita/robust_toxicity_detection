* re-import the data
egen err = group(quad)
gen nonas=(non_ascii=="TRUE")
gen ownl=strmatch(target, labels)
gen labflag = (labels~="")
gen other_labels=(labflag==1 & ownl==0)
label define error_labs 1 "FN" 2 "FP" 3 "TN" 4 "TP"
label values err error_labs
gen positive = (quad=="TP" | quad=="FN")
gen pred_positive = (quad=="TP" | quad=="FP")

gen lab_toxic=strmatch(labels,"*toxic*")
gen lab_severe_toxic=strmatch(labels,"*severe_toxic*")
gen lab_threat=strmatch(labels,"*threat*")
gen lab_obscene=strmatch(labels,"*obscene*")
gen lab_insult=strmatch(labels,"*insult*")
gen lab_identity_hate=strmatch(labels,"*identity_hate*")
summ lab_*
gen lab_toxic_all =(lab_toxic | lab_severe_toxic)

mlogit err oov_ratio len i.nonas  i.repeats i.casing i.curtsey i.lab_threat i.lab_obscene i.lab_insult i.lab_identity_hate /*
             			*/  if target=="severe_toxic" | target=="toxic", base(3)

margins, dydx(*) 


mlogit err oov_ratio len i.nonas  i.repeats i.casing i.curtsey /*
          */ i.lab_insult i.lab_obscene i.lab_identity_hate i.lab_toxic_all if target=="threat" , base(3)
		  
predict fn if target=="threat", outcome(1)
predict fp if target=="threat", outcome(2)
predict tn if target=="threat", outcome(3)
predict tp if target=="threat", outcome(4)

egen pattern=group(oov_ratio len nonas  repeats casing curtsey /*
          */ lab_insult lab_obscene lab_identity_hate lab_toxic_all) if target=="threat"
		  
mlogit err oov_ratio len i.nonas  i.repeats i.casing i.curtsey /*
          */ i.lab_insult i.lab_obscene i.lab_identity_hate i.lab_toxic_all /*
		  */ if pattern~=142 & pattern~=130 & pattern~=88 & target=="threat" , base(3)
          
margins, dydx(*) 

foreach v in  nonas  repeats casing curtsey /*
          */ lab_insult lab_obscene lab_identity_hate lab_toxic_all {
		  tab quad `v' if pattern~=142 & pattern~=130 & pattern~=88 & target=="threat"
		  }



mlogit err oov_ratio len i.nonas  i.repeats i.casing i.curtsey /*
       */i.lab_threat i.lab_obscene i.lab_identity_hate i.lab_toxic_all if target=="insult", base(3) 

margins, dydx(*) 

mlogit err oov_ratio len i.nonas  i.repeats i.casing i.curtsey /*
           */i.lab_threat i.lab_insult i.lab_identity_hate i.lab_toxic_all if target=="obscene", base(3)
margins, dydx(*) 

mlogit err oov_ratio len i.nonas  i.repeats i.casing i.curtsey /*
         */i.lab_threat i.lab_insult i.lab_obscene i.lab_toxic_all if target=="identity_hate", base(3)
margins, dydx(*) 
