real=[];dest=[];joint=[];references=[];denoms=[];records=[]
for t in lock['templates']:
 gen,learner,S=t['generator'],t['learner'],t['seed_n'];seeds=t['seed_membership']
 sub=c[(c.generator==gen)&(c.learner==learner)].copy()
 check(sorted(sub.seed.unique())==seeds and sorted(sub.hospital_id.unique())==ids,'Seed/hospital namespace mismatch')
 check(set(zip(sub.seed,sub.hospital_id))=={(s,h) for s in seeds for h in ids},'Incomplete Cartesian matrix')
 matrix=sub.pivot(index='seed',columns='hospital_id',values='EUL').reindex(index=seeds,columns=ids).to_numpy()
 ref=tb[(tb.generator==gen)&(tb.learner==learner)].iloc[0]
 check(ref.hospital_n==85 and ref.seed_n==S and ref.cell_n==S*85 and ref.converged,'Target B row denominator/status mismatch')
 lower=float(ref.new_seed_new_hospital_PI95_lower);upper=float(ref.new_seed_new_hospital_PI95_upper)
 check(np.isfinite([lower,upper]).all() and lower<=upper,'Invalid Target B PI')
 total_real=85*S*(S-1)//2;total_dest=S*85*84//2
 for delta in lock['delta_grid']:
  q=matrix<=delta;k=q.sum(axis=0);j=q.sum(axis=1)
  rnum=k*(S-k);dnum=j*(85-j);rden=S*(S-1)//2;dden=85*84//2
  # Internal QC re-counts unordered pairs by explicit upper-triangle XOR, not k*(S-k).
  xor_seed=np.logical_xor(q[:,None,:],q[None,:,:]);ix=np.triu_indices(S,k=1)
  check(np.array_equal(xor_seed[ix[0],ix[1],:].sum(axis=0),rnum),'Seed pair enumeration mismatch')
  xor_h=np.logical_xor(q[:,:,None],q[:,None,:]);ixh=np.triu_indices(85,k=1)
  check(np.array_equal(xor_h[:,ixh[0],ixh[1]].sum(axis=1),dnum),'Hospital pair enumeration mismatch')
  rrev=int(((k>0)&(k<S)).sum());drev=int(((j>0)&(j<85)).sum())
  rq=int((k==S).sum());rn=int((k==0).sum());dq=int((j==85).sum());dn=int((j==0).sum())
  check(rrev+rq+rn==85 and drev+dq+dn==S,'Reversal/unanimity denominator partition')
  rrate=rnum/rden;drate=dnum/dden
  common={'generator':gen,'learner':learner,'delta':delta}
  rr=common|{'hospitals_evaluable':85,'seed_count':S,'total_seed_pairs':total_real,'discordant_seed_pairs':int(rnum.sum()),'realization_pair_discordance_rate':float(rnum.sum()/total_real),'hospital_rate_distribution_json':json.dumps([{'hospital_id':int(h),'qualified_count':int(k[z]),'pair_denominator':rden,'discordant_pair_count':int(rnum[z]),'pair_discordance_rate':float(rrate[z])} for z,h in enumerate(ids)]),'hospital_reversal_count':rrev,'hospital_reversal_proportion':rrev/85,'unanimous_qualified_hospital_count':rq,'unanimous_qualified_hospital_proportion':rq/85,'unanimous_not_qualified_hospital_count':rn,'unanimous_not_qualified_hospital_proportion':rn/85}
  dr=common|{'estimable_seed_count':S,'hospital_count':85,'total_hospital_pairs':total_dest,'discordant_hospital_pairs':int(dnum.sum()),'destination_pair_discordance_rate':float(dnum.sum()/total_dest),'seed_rate_distribution_json':json.dumps([{'seed':int(s),'qualified_count':int(j[z]),'pair_denominator':dden,'discordant_pair_count':int(dnum[z]),'pair_discordance_rate':float(drate[z])} for z,s in enumerate(seeds)]),'seed_reversal_count':drev,'seed_reversal_proportion':drev/S,'unanimous_qualified_seed_count':dq,'unanimous_qualified_seed_proportion':dq/S,'unanimous_not_qualified_seed_count':dn,'unanimous_not_qualified_seed_proportion':dn/S}
  for row,rates,prefix in [(rr,rrate,'hospital_rate'),(dr,drate,'seed_rate')]:
   for label,val in zip(['min','Q1','median','Q3','max'],np.quantile(rates,[0,.25,.5,.75,1],method='linear')):row[prefix+'_'+label]=float(val)
  real.append(rr);dest.append(dr)
  qualified=int(q.sum());non=S*85-qualified
  jr=common|{'total_cells':S*85,'qualified_cells':qualified,'not_qualified_cells':non,'qualified_fraction':qualified/(S*85),'both_states_present':qualified>0 and non>0,'hospital_reversal_count':rrev,'hospital_reversal_proportion':rrev/85,'seed_reversal_count':drev,'seed_reversal_proportion':drev/S};joint.append(jr)
  if upper<=delta:state='FULL_TARGET_B_QUALIFIED';opposite=non
  elif lower>delta:state='FULL_TARGET_B_NOT_QUALIFIED';opposite=qualified
  else:state='FULL_TARGET_B_INDETERMINATE';opposite=None
  references.append(common|{'reference_lower_PI':lower,'reference_upper_PI':upper,'full_Target_B_state':state,'total_observed_cells':S*85,'qualified_cell_count':qualified,'not_qualified_cell_count':non,'qualified_cell_fraction':qualified/(S*85),'not_qualified_cell_fraction':non/(S*85),'opposite_cell_count':opposite,'CELL_TO_TARGET_B_DISCORDANCE':opposite/(S*85) if opposite is not None else None,'discordance_applicability':'DEFINITIVE_REFERENCE' if opposite is not None else 'NOT_APPLICABLE_INDETERMINATE_REFERENCE'})
  out=sub[['generator','learner','seed','hospital_id','EUL']].copy();out['delta']=delta;out['binary_decision']=np.where(out.EUL<=delta,'QUALIFIED','NOT_QUALIFIED');out['utility_estimable']=True;out['eligibility_scope']='PRIMARY_200_20_20';records.append(out)
  denoms.append(common|{'hospital_count':85,'hospital_identity_sha256':lock['hospital_identity_sha256'],'generation_attempts':15,'utility_estimable_seed_count':S,'collapsed_attempt_count':15-S,'collapsed_seed_ids':';'.join(map(str,collapsed_seed_ids)) if gen=='TabDDPM' else '', 'utility_seed_ids':';'.join(map(str,seeds)),'expected_cells':S*85,'observed_cells':len(sub),'missing_expected_cells':0,'duplicate_keys':0,'nonfinite_EUL':0,'unregistered_seed_cells':0,'collapsed_seed_cells':0,'excluded_nonprimary_hospital_cells':int(((cells.generator==gen)&(cells.learner==learner)&~cells.hospital_id.isin(ids)).sum()),'total_seed_pairs':total_real,'total_hospital_pairs':total_dest,'denominator_boundary':'Conditional on utility-estimable generation; no collapsed attempt is assigned a binary state'})
per=pd.concat(records,ignore_index=True)
check(len(per)==9520 and not per.duplicated(['generator','learner','seed','hospital_id','delta']).any(),'Saved decision matrix grain/count')
per.to_parquet(OUT/'U6_02_PER_CELL_DECISIONS.parquet',index=False,compression='zstd')
realization=pd.DataFrame(real);destination=pd.DataFrame(dest);mat=pd.DataFrame(joint);reference=pd.DataFrame(references);reference['opposite_cell_count']=reference.opposite_cell_count.astype('Int64')
for frame,name in [(realization,'U6_02_REALIZATION_INSTABILITY.csv'),(destination,'U6_02_DESTINATION_INSTABILITY.csv'),(mat,'U6_02_JOINT_MATRIX_SUMMARY.csv'),(reference,'U6_02_TARGET_B_REFERENCE_CONCORDANCE.csv'),(pd.DataFrame(denoms),'U6_02_DENOMINATOR_REPORT.csv')]:
 check(len(frame)==8,'All eight corrected TabDDPM analyses required');frame.to_csv(OUT/name,index=False,float_format='%.17g')
