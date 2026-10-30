from pathlib import Path
import json,hashlib,datetime,platform
H=Path(__file__).resolve().parent;r=json.loads((H/'QUANTITATIVE_COMPARISON.json').read_text());c=json.loads((H/'RESOURCE_LEDGER.json').read_text());src=json.loads((H/'AUTHOR_TEXT_SOURCE_MANIFEST.json').read_text());inputs=json.loads((H/'INPUT_VERIFICATION.json').read_text())+json.loads((H/'COMPATIBLE_CHECKPOINT_VERIFICATION.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
source={'repo':'https://github.com/tiwarylab/ExpTM','commit':'28f1252eed7cbaaf67d901b337ffee1885552fbe','commit_date':'2026-08-12T21:07:32Z','read_UTC':utc,'source_selection':'unmodified selected GC_Lattice_gas author code; no older thermomaps-ising execution','GitHub_license_metadata':None,'LICENSE_file_in_recursive_tree':False,'setup_classifier':'License :: OSI Approved :: MIT License','licensing_interpretation':'MIT classifier found, but no full repository license text recovered; do not attribute an unverified repository license. Private research snapshot retains author provenance.','article':'https://doi.org/10.1038/s41467-026-77612-y','article_state':'Article in Press, published14September2026; downloaded identified primary PDF and14-page supplement','linked_arxiv':'https://arxiv.org/abs/2503.15080','colab_link':'https://colab.research.google.com/drive/1Q5-LyumxZfZc6aHt71ZKOsGWfjrGEEzV?usp=sharing','colab_access':'link verified in README; contents unavailable through public lookup in this task; no login or remote Colab execution','source_text_hashes':src,'input_byte_hashes':inputs}
(H/'SOURCE_PIN_AND_LICENSE.json').write_text(json.dumps(source,indent=2)+'\n')
statuses={'environment_installed':True,'selected_author_core_executes':True,'author_checkpoint_evaluated':True,'checkpoint_based_not_independent_training':True,'quantitative_published_result_reproduced':'UNRESOLVED','independent_training_reproduced':False,'parameter_generalization_demonstrated':False,'computational_advantage_demonstrated':False,'ABP_mapping_demonstrated':False}
manifest={'task':'EXPTM-AUTHOR-CODE-REPRODUCTION-v1','status':'COMPLETED_SCOPED_RELEASE','strongest_supported_outcome':'AUTHOR_CHECKPOINT_EXECUTION_VERIFIED_WITH_LIMITED_QUANTITATIVE_DIAGNOSTIC','commit':source['commit'],'example':'GC_Lattice_gas only','checkpoint':inputs[-1],'configuration':'FROZEN_TARGET.json','execution_route':'unmodified author core / literal GC_ising_code input recipe / CPU checkpoint loading; no retraining','outcomes':statuses,'quantitative_comparison':r,'registered_quantitative_target':'single intended Fig2(b) symmetry-center density, native controlsT5/P0; full physical parameter/checkpoint bindings remain unverified','reference':'analytic0.5 under intended finite periodic grand-canonical particle-hole symmetry; no digitized or original MC cell treated as reference','resource_ledger':'RESOURCE_LEDGER.json','measured_cost_at_report_creation':{'wall_seconds':c['measured_wall_seconds'],'CPU_seconds':c['measured_CPU_seconds'],'GPU_seconds':0},'new_training_updates':0,'stochastic_physical_trajectories':0,'new_ABP_steps':0,'attempts':[{'stage':'inspect_checkpoint','outcome':'FAILED_STRICT_LOAD model49, fourchannels/dim16 versus fixed3channel/dim32; preserved logs'},{'stage':'inspect_compatible_checkpoint','outcome':'strict model1 loadPASS; selection made solely from compatibility before generation'},{'stage':'smoke_checkpoint','outcome':'2 finite samples,100 reverse steps'},{'stage':'quantitative_checkpoint','outcome':'256 finite samples,16 fixed batches; all retained'}],'compatibility':'COMPATIBILITY_RECORD.md; no author file patched','one_next_step':'Recover a figure-bound author run bundle: three-channel checkpoint from the published10epoch/T7 run, exact T/mu normalization, training subset/seeds and numeric MC/generated Fig2 density arrays. No further generation/retraining authorized automatically.','successor_executed':False,'historical_STOPS_unchanged':['CP4 STOP','Track B STOP','Research C NO-GO','Research D allocation-only STOP','G1 reference limitations and all earlier failed mechanisms'],'protected_Pe_access':False}
(H/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
report=f'''# ExpTM author-code reproduction v1

**Strongest outcome: author checkpoint execution verified, with one limited quantitative density diagnostic. Full published-result reproduction remains UNRESOLVED.**

## Source, selected workflow and scope

Pinned [tiwarylab/ExpTM](https://github.com/tiwarylab/ExpTM/tree/{source['commit']}) at `{source['commit']}`. Actual application directory is `GC_Lattice_gas/expTM_data_genenration`; its spelling is retained. Core `tm/` is present. README's root installation path is not an existing complete environment. CO2 assets and the Colab link exist, but neither application nor remote Colab ran. The older thermomaps-ising and later HOOMD tutorial were not substituted.

We executed the authors' lattice-gas preprocessing/sampling recipe from GC_ising_code.ipynb cells1–7 and9–10 through a small local runner, replacing training cell8 with supplied checkpoint loading. No author source code was patched. All source/data/checkpoint bytes are Git-blob and SHA-256 verified. License metadata is null and no LICENSE file is present; setup.py advertises an MIT classifier but a complete repository license was not recovered. See SOURCE_PIN_AND_LICENSE.json.

| Outcome | Established? |
|---|---|
| CPU environment installed | Yes; Python3.12 / torch2.5.1+cpu; exact dependencies supplied |
| Selected code executes end to end | Yes;100-step smoke,2 finite20×20 outputs |
| Author checkpoint evaluated | Yes; model_1.pt, strict3-channel dim32 load |
| Published Fig2 result reproduced | **Unresolved configuration/reference binding** |
| Independent retraining reproduced | No;0 new optimizer updates |
| Generalization / total efficiency demonstrated | No |

## One frozen comparison

The intended paper link is the symmetry-center density in Fig2(b), physical μ=−8 and intendedT=5. We actually evaluated literal author notebook controls `temperature=5, pressure=0`. The normalized mean-control endpoints are−1/+1; interpreting zero as the physical symmetry center is plausible from the stated anchors but is not a recovered figure-run binding.

Original data contain5000 configurations per supplied anchor, shape(5000,2,20,20), float64, spin-like values±1 and stored label±4. The literal notebook prepares(10000,3,20,20): spin,temperature5,normalized mean-control−1/+1. Author dequantization0.01,dim32 U-Net,nearest-exact20↔32 interpolation,polynomial100-step VP diffusion,eta1,gamma1 and zero-threshold occupancy decoding were retained. Model-level self_condition is its defaultFalse; backbone self_condition=True/eval_mode=train is preserved. Only one condition was generated, with locally frozen seeds; no hyperparameter or checkpoint performance search.

For the intended periodic lattice-gas Hamiltonian, complementing every occupancy at μ=−εz/2 gives equal equilibrium weight to n and1−n. Thus Eρ=1/2 for any positiveT. This exact physical expectation, not a digitized figure or a guessed MC cell, is the comparison reference. The reference does not resolve checkpoint parameter units or validate a learned distribution merely by its mean.

## Actual quantitative output

| Quantity | Result |
|---|---:|
| Generated configurations |256, all finite and retained|
| Output array shape |(256,3,20,20)|
| Mean decoded density |{r['density_mean']:.17g}|
| Exact symmetry reference, conditional mapping |0.5|
| Absolute difference |{r['absolute_difference']:.17g}|
| Sample density SD |{r['density_sd']:.17g}|
| Descriptive conditional SEM |{r['descriptive_conditional_SEM']:.17g}|
| Prospective conditional Hoeffding95% enclosure |[{r['Hoeffding95_interval'][0]:.17g},{r['Hoeffding95_interval'][1]:.17g}]|
| Distinct decoded configurations |{r['unique_thresholded_configurations']}/256|

The exact reference lies inside the predeclared bounded enclosure. That enclosure is deliberately conservative and **does not certify a small accuracy tolerance**. SEM is descriptive conditional sampling information; no normal-based CI or post-result acceptance threshold was introduced. One fitted checkpoint and one preparation/generation seed provide no retraining uncertainty. Generated configurations, not400 sites, are the sampling units; the bound assumes independent idealized random draws conditional on the fixed model. It is not a physical-reference-history or parameter-mapping uncertainty bound. A correct central mean is a weak test and does not validate correlations, occupancy fluctuations, phase boundaries or thermodynamic transport.

![Registered density comparison](density_comparison.png)

Firstsix draws, prospectively fixed indices, are shown in first_six_generated_lattices.png; all256 densities and raw outputs are saved.

## Why the published target is not yet bound

| Concrete finding | Consequence |
|---|---|
| model49 has4-channel/dim16 weights; strict loading failed | It is incompatible with the specified lattice-gas network. Failure logs are retained; no resizing or new representation. |
| model1 matches3-channel/dim32, epoch1; optimizer steps64 | Real author checkpoint, but not demonstrably the published training run. |
| Current notebooks useT1 orT5; paper trainsT7 | No unverified temperature conversion or scientific correction was invented. |
| Provided data5000+5000; published SI says10000+10000 | The exact training subset for the figure is not recovered. |
| Published supplement specifies10epochs/100 diffusion steps; checkpoint lacks schedule/scaler/seeds |100steps agrees with the public recipe, but training provenance is incomplete. |
| Provided density_heatmap.npy is5×5 without axes/checkpoint/seed bindings | It is not treated as the published MC/generated numerical map. |

The [published Article in Press](https://doi.org/10.1038/s41467-026-77612-y), Results/Fig2 and Methods Eqs18–21, and [published supplement](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-026-77612-y/MediaObjects/41467_2026_77612_MOESM1_ESM.pdf), Notes1–3/Algorithm1, were inspected along with the README-linked arXiv version. The published calculation uses twoT7 anchors and evaluates density across a two-variable grid. This release tests neither that full grid nor its timing comparison. Different hardware, sample counts and uncharged original training-data acquisition prevent a speedup inference from our CPU timing.

## Cost, handoff and STOP

Quantitative checkpoint execution measured439.31989176000025wall/874.973192322CPU seconds, including preparation/loading; sampling loop alone436.0686283489995wall seconds. Cumulative setup, transfer, failed inspection, smoke, generation, analysis and verification costs are in RESOURCE_LEDGER.json; later publication costs are separate. All computation remains below1800 activewall/7200 aggregateCPU ceilings. GPU0,paidcompute0,clusterjobs0,newtraining0. The initial15s inspection allowance is a labelled conservative budget reserve, not a measurement. Overlapping setup/download costs are conservatively summed; the ledger race was corrected from its exact metering receipt, never by removing costs.

**One justified next step:** obtain the author figure-bound run bundle specifying theT7/10epoch checkpoint, exact control transformations, training-data selection and numeric Fig2 reference arrays. This is a concrete provenance prerequisite before a full result comparison, not permission to retrain or guess a new configuration. No successor is executed.

LEARNING_HANDOFF.md supplies the executed function sequence, shapes, paper links and an optional one-sample rerun. COORDINATION_HANDOFF.md separates technical execution, conditional density evidence and unresolved scientific claims. Historical ABP/CP4/TrackB/C/D decisions and locked Pe7.5/22.5/27.5 roles remain unchanged. This equilibrium-system exercise establishes no activity-to-distribution mapping for interacting ABPs. STOP.
'''
(H/'REPRODUCTION_REPORT.md').write_text(report)
(H/'COORDINATION_HANDOFF.md').write_text(f'''# Coordination — completed ExpTM author-code release

**AUTHOR_CHECKPOINT_EXECUTION_VERIFIED_WITH_LIMITED_QUANTITATIVE_DIAGNOSTIC.**

- [x] Actual ExpTM source pinned at `{source['commit']}`; selected author files/data byte-verified.
- [x] CPU environment installed;100-step end-to-end smoke complete.
- [x] Compatible author model1 strictly loaded and evaluated,0 new training updates.
- [x] One nativeT5/mean-control0 condition,256 complete configurations; meanρ={r['density_mean']:.17g}, absolute difference from the conditional exact symmetry expectation0.5={r['absolute_difference']:.17g}.
- [x] All outputs/RNG states, failed model49 compatibility attempt, resource costs and commands retained.
- [ ] Published Fig2 training/scaling/checkpoint binding remains unresolved; no full numerical reproduction claim.
- [ ] No independent training, parameter-generalization, phase-boundary or speedup benchmark.

The prospective conditional Hoeffding enclosure contains0.5, but its width is too broad to certify a tight practical accuracy target. SEM is descriptive only. One central density mean cannot validate the joint spatial law. Metadata mismatch is not a demonstrated ExpTM method failure.

The smallest next action is to recover one figure-bound author run package: three-channelT7/10epoch checkpoint, exact parameter normalization and training subset/seeds, plus original numeric MC/generated Fig2 densities. Do not launch training to fill this provenance gap, vary controls to look for agreement, or substitute another system. No author correspondence or GitHub issue was sent. No automatic successor.

This saved artifact is the handoff to Coordination and the learning chat; direct access or communication between chats is not assumed. All historical STOP/NO-GO/reference-limited outcomes remain attached to their original investigations. Protected ABP activities were untouched. See RESOURCE_LEDGER.json for final measured totals and administrative reserve.
''')
(H/'REFERENCE_IDENTITY_AND_LIMITS.md').write_text(r'''# One exact density reference; scope of the claim

Let M=400 sites, z=4 neighbors, B=zM/2 undirected edges, and n_i in{0,1}. For the intended Hamiltonian E(n)=−ε sum_edges n_i n_j−μ sum_i n_i,

E(1−n)−E(n)=−εB−μM+(εz+2μ)sum_i n_i.

At μ=−εz/2 both the constant and the n-dependent term vanish. The finite periodic equilibrium law at anyT>0 is therefore invariant under n→1−n, and Eρ=1−Eρ=1/2. With the intended ε=4 this center isμ=−8, matching the paper's stated center. This standard symmetry argument is not a new theorem.

The numerical decoder is exactly the author's threshold recipe: n_i=1 when generated channel0≥0,0 otherwise;ρ=sum n_i/400. This decoder is a generation convention and does not itself identify a true grand-canonical law.

Native mean-control0 is centered between the normalized training labels−1/+1. Associating it with physicalμ=−8 is a conditional intended mapping, not a verified checkpoint-to-paper transform. Temperature5 is the literal GC_ising_code notebook input, not a reconstructedT7 normalization. The symmetry reference isT-independent, but that does not prove that the supplied network conditions have physical units.

For N=256 generated densities bounded in[0,1], idealized independent sampling conditional on the fitted network gives P(|rho_bar−E_model rho|>sqrt(log(40)/(2N)))≤0.05. The bound targets E_model rho, not E_physics rho; it says nothing about unknown training seeds, physical sampling histories, normalization or scientific protocol error. Descriptive SEM uses between-generated-configuration variability; sites are not separate replicates. No post-result Gaussian interval or new closeness threshold is added.

Compatibility with0.5 through the conservative bound does not establish practical equality. A model with incorrect spatial correlations or no useful condition dependence can still pass this one weak moment check. Neither a compatible nor incompatible mean alone establishes the success or failure of the published expTM method.
''')
print(manifest['strongest_supported_outcome'])
