"""Fabricated mathematical tests; no clinical data, production fits or simulations."""
import hashlib,math
import numpy as np
from scipy.stats import norm
from decision_evaluation.core import (derive_seed,standardized_draws,probability_state,interval_state,summarize_decisions,BalancedCrossedREML,CONFIDENCE_GRID,DELTAS)
from decision_evaluation import simulation,summarize,bootstrap,operational

def test_decision_cutoffs_and_central_interval_boundary():
    assert probability_state(.95)=='QUALIFIED'
    assert probability_state(.04)=='NOT_QUALIFIED'
    assert probability_state(.5)=='INDETERMINATE'
    assert interval_state(.025,.10,.10)=='QUALIFIED'
    assert interval_state(.05,.15,.05)=='INDETERMINATE'

def test_selective_metric_denominators():
    r=summarize_decisions([{'oracle_state':'QUALIFIED','estimated_state':'QUALIFIED'},{'oracle_state':'INDETERMINATE','estimated_state':'QUALIFIED'},{'oracle_state':'NOT_QUALIFIED','estimated_state':'INDETERMINATE'}])
    assert r['evaluable_n']==3 and r['definitive_n']==2
    assert np.isclose(r['DER'],2/3) and np.isclose(r['DY'],2/3)
    assert np.isclose(r['SER'],.5) and np.isclose(r['FQ_all'],1/3)
    assert np.isclose(r['FQ_def'],.5) and np.isclose(r['CDY'],1/3)

def test_fixed_grids():
    assert CONFIDENCE_GRID==(.80,.90,.95,.975,.99)
    assert DELTAS==(.025,.05,.10,.20)

def test_random_stream_derivation_matches_frozen_key():
    import decision_evaluation.core as c
    k=f'{c.SEED_PREFIX}|fixture|G|fabricated|mixed|1|20261020'
    assert derive_seed('fixture','G','fabricated','mixed',1)==int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],'big')

def test_standardized_t5_exact_scaling():
    actual=standardized_draws('T',20,np.random.default_rng(42))
    expected=np.random.default_rng(42).standard_t(5,20)*math.sqrt(3/5)
    np.testing.assert_array_equal(actual,expected)

def test_skew_normal_analytic_centering_scaling():
    rg=np.random.default_rng(42);d=5/math.sqrt(26)
    raw=d*abs(rg.standard_normal(20))+math.sqrt(1-d*d)*rg.standard_normal(20)
    expected=(raw-d*math.sqrt(2/math.pi))/math.sqrt(1-2*d*d/math.pi)
    np.testing.assert_array_equal(standardized_draws('S',20,np.random.default_rng(42)),expected)

def test_balanced_likelihood_matches_dense_fabricated_covariance():
    v=np.array([[[.002,.0002],[.0002,.003]],[[.004,.0005],[.0005,.002]],[[.003,.0001],[.0001,.005]]])
    y=np.array([[.01,.03],[.04,.02],[.01,.02]]);theta=np.log([.001,.002,.0015]);s,h,i=np.exp(theta)
    cov=np.zeros((6,6))
    for a in range(3):cov[a*2:a*2+2,a*2:a*2+2]=v[a]+h*np.ones((2,2))+i*np.eye(2)
    cov+=s*np.tile(np.eye(2),(3,3));iv=np.linalg.inv(cov);yy=y.ravel();one=np.ones(6);mu=one@iv@yy/(one@iv@one);res=yy-mu
    expected=.5*(np.linalg.slogdet(cov)[1]+np.log(one@iv@one)+res@iv@res+5*np.log(2*np.pi))
    assert abs(BalancedCrossedREML(list(v)).objective(theta,y)-expected)<1e-12
