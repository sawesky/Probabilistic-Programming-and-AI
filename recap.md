Najveci problem je bila fja za semplovanje `y`:
U mnistu su grayscale slike pa moze da se radi bernulijeva raspodela dist.Bernoulli, znaci imali smo (ovo je u modelu poslednjih par linija) 
```
pyro.sample("y",
            dist.Bernoulli(mask_loc, validate_args=False).to_event(1),
            obs=mask_ys,
            )
```

pa je isao BCE loss, a kod CIFARa ne moze to nego se radi NLL loss (negative log likelihood) sto je zapravo generalizacija BCE, a NLL mozemo da koristimo i za Normalnu raspodelu (pyro sam to under the hood zameni ali postoje velike razlike u rezultatima u onom nasem kodu za baseline gde je definisan maskedBCEloss, pa sam dodao da ima i maskedNLLloss)

```
pyro.sample("y",
            dist.Normal(mask_loc, 0.05).to_event(1),
            bs=mask_ys,
            )
```

Sad bi trebalo da je sve lepo poklopljeno

Postoje neke razlike kad se pusti recimo sa i bez `y_hat`

Ne znam koliko bi trebalo da se cimamo oko "data science" tipa predstavljanja projekta jer pise tamo da se nadje jedan inference algoritam koji radi (to pise na sajtu ):

Scope

If you pick a hard problem, then all of your effort will probably go towards finding one model and implementing one sophistaced inference algorithm that solves it. For such problems, document and share the process of finding a model / inference algorithm combination that made it work.

Rezultati u res, gledati samo slike ove brojke moraju biti ispravljene, same stvari oko modela se nisu mnogo promenile

Nije nesto mnogo kod diran


Ovo za probati jos:
spline / conditionalarnn, flow difference

moze vec prez da se radi