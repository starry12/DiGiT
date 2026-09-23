# Publishing the submission

This repository starts with one initial commit and contains no parent research Git history. Publish this directory, not the research workspace or the preserved broad artifact repository. `git status --short` should be clean; `bash run.sh verify` checks the source snapshot.

Create an empty repository with the intended visibility, then set its supplied URL:

```bash
git remote add origin <REPOSITORY_URL>
git push -u origin main
git push origin ae-pa-v1
```

Provide the repository URL and immutable tag/commit in the AE submission, along with private server access instructions. A later addition should use a new commit/tag and updated scope; retain the submitted tag so reviewers can reproduce the original version. Do not commit datasets, credentials, checkpoints, local deployment bindings or generated results.

The project license decision is pending in [LICENSE_STATUS](../LICENSE_STATUS.md). GitHub publication and conference-form submission are separate actions; neither is implied by creating this local repository.
