# Audit des écarts — intégration RATISS × LeWM

## Conclusion

Le run Colab terminé ne correspond pas à l’intégration totale RATISS attendue. Il s’agit d’un entraînement LeWM sur fenêtres PushT auquel une pénalisation thermodynamique proxy a été ajoutée dans la boucle de perte. Les fichiers `.py` RATISS n’ont pas été utilisés comme couche d’orchestration complète, et `P_sig` n’a pas été évalué avec une loi dépendante de l’environnement.

La conclusion précédente doit donc être requalifiée : **le run A/B est un test exploratoire de pénalisation proxy, pas une validation de RATISS intégré au système World Model + environnement**.

## Écarts constatés

| Composant attendu | État dans le dépôt | État dans le notebook exécuté |
|---|---|---|
| `TopologicalInvariantPlugin` | Mesure postérieure avec `ripser`, non différentiable | Utilisé uniquement pour logging |
| Loi `P_sig` dépendante de l’environnement | Absente comme interface explicite dans les `.py` publiés | Non utilisée |
| `ThermodynamicRegulator` | Mesure et pénalité scalaire, poids nul par défaut | Remplacé par une pénalisation de variance écrite directement dans le notebook |
| `CohesionMonitor` | Corrélation générique sur embeddings | Utilisé pour logging, sans état environnemental |
| `RATISSHookBundle` | `post_encode` et `post_predict` disponibles | Non utilisé |
| `attach_hooks` | Installe seulement un hook encoder qui retourne la sortie inchangée | Non utilisé ; aucun hook effectif dans la boucle |
| `InvariantCache` | Cache conditionné par `min_psig` | Non utilisé |
| Environnement PushT | Frontière documentée côté `stable-worldmodel` | Aucune boucle `reset/step` ni rollout évalué |
| A/B LeWM complet | Non modifiant upstream, possible via harness | Notebook modifie seulement la perte locale du training loop |

## Pourquoi `P_sig` n’est pas encore la loi attendue

Dans `ratiss_topo_plugin.py`, `P_sig` est actuellement calculé comme la durée de vie maximale d’une classe H1 dans une homologie persistante appliquée au nuage latent d’une fenêtre. La fonction reçoit seulement `embeddings` et ne reçoit ni état de l’environnement, ni géométrie, ni contraintes, ni action, ni événement physique.

Cette interface ne peut donc pas exprimer une loi du type : « dans l’environnement PushT, telle configuration géométrique ou telle transition doit être considérée comme cohérente ». Elle mesure une structure topologique latente générique. Elle ne sait pas si une boucle est physiquement valide, invalide, utile ou causée par un artefact de représentation.

De plus, le seuil calibré dans les notes d’injection n’est pas codé comme une politique environnementale dans le plugin. La calibration historique mentionne un seuil exploratoire d’environ 15 % de la référence, mais aucune classe `EnvironmentLaw`, aucun seuil par environnement et aucune association entre `P_sig` et les transitions PushT n’existent dans l’interface actuelle.

## Ce que le notebook a réellement optimisé

Le bras RATISS du notebook ajoute :

```python
loss = mse + THERMO_WEIGHT * mean((pred - mean(pred))**2)
```

Ce terme est une pénalisation de variance des prédictions. Il ne correspond pas à `TopologicalInvariantPlugin.loss()`, qui est détachée et non différentiable, et il ne constitue pas une exécution de `RATISSHookBundle.post_predict()`.

Le notebook entraîne directement sur des fenêtres pré-calculées. Il ne fait pas interagir le modèle avec l’environnement PushT. Il ne vérifie donc ni la cohérence entre l’état latent et l’état physique, ni la qualité d’un rollout, ni l’effet de RATISS sur une décision ou une action.

## Architecture complète à implémenter

L’intégration correcte doit rester externe à LeWM, mais relier les trois frontières suivantes :

1. **Frontière encodeur** : récupérer `emb` après `JEPA.encode()`.
2. **Frontière prédiction** : récupérer `pred_emb` et l’action prévue après `JEPA.predict()`.
3. **Frontière environnement** : récupérer l’état PushT avant et après `env.step(action)`, ainsi que `reward`, `done` et les informations géométriques disponibles.

Une classe d’orchestration dédiée doit recevoir explicitement :

```python
law.evaluate(
    env_name="pusht",
    state_t=state_t,
    action_t=action_t,
    state_tp1=state_tp1,
    embedding_window=emb,
    predicted_embedding=pred,
)
```

Elle doit retourner au minimum :

```python
{
    "p_sig": ...,
    "p_sig_threshold": ...,
    "topology_valid": ...,
    "physical_consistency": ...,
    "cohesion": ...,
    "thermo_drift": ...,
    "cache_accepted": ...,
}
```

Le plugin topologique peut rester non différentiable et servir de **contrôleur, filtre, score ou signal de logging**. Si une influence directe sur l’apprentissage est nécessaire, elle doit passer par un surrogate différentiable explicitement défini et comparé à une version sans surrogate.

## Protocole corrigé

Le prochain protocole ne doit pas comparer « Vanilla contre une autre perte » comme résultat principal. Il doit comparer :

- **LeWM vanilla + même politique d’évaluation** ;
- **LeWM + couche RATISS d’observation et de contrôle** ;
- puis, séparément, **LeWM entraîné avec un surrogate RATISS** si ce surrogate est validé.

Les deux premiers bras doivent mesurer :

- MSE d’embeddings ;
- `P_sig` conditionné par l’environnement ;
- cohérence physique ;
- reward et taux de succès PushT ;
- stabilité des rollouts ;
- décisions d’acceptation/rejet du cache invariant ;
- dérive thermodynamique ;
- coût de calcul.

Une baisse de `P_sig` n’est pas automatiquement positive. Le critère doit être la cohérence entre la signature, la loi environnementale et la performance de rollout.

## Informations manquantes pour finaliser la loi

Le dépôt actuel ne contient pas la définition de la loi environnementale mentionnée par le besoin. Pour l’implémenter sans l’inventer, il faut fournir ou préciser :

- les variables physiques PushT considérées comme cohérentes ;
- la condition qui rend une boucle topologique valide ou invalide ;
- le seuil ou la règle de décision par environnement ;
- la conséquence attendue de `P_sig` : logging, rejet d’action, régulation, cache ou terme d’apprentissage ;
- le lien exact entre les états PushT du dataset et les états retournés par l’environnement pendant un rollout.

## Décision

Les résultats du notebook doivent rester dans les archives comme **expérience proxy**. Ils ne doivent pas être présentés comme la validation complète RATISS. La prochaine implémentation doit d’abord construire l’adaptateur environnemental et utiliser réellement `RATISSHookBundle`, `InvariantCache` et les plugins `.py` dans un harness de rollout contrôlé.

## Références

[1]: https://github.com/samajonathan9-source/ratiss-lewm-integration "Dépôt d’intégration RATISS × LeWM"
[2]: https://github.com/lucas-maes/le-wm "Dépôt officiel LeWorldModel"
[3]: https://huggingface.co/datasets/lerobot/pusht "Dataset officiel LeRobot PushT"
