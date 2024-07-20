from pathlib import Path

import git

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def git_repo_query(code_root: Path) -> git.Repo:
    # Try to get repo object
    try:
        repo = git.Repo(str(code_root))
    except git.exc.InvalidGitRepositoryError:
        log.warning("No git repo found")
        return None
    # Try to get branch name
    try:
        branch_name = repo.active_branch.name
    except TypeError as e:
        if repo.head.is_detached:
            branch_name = "DETACHED_HEAD"
        else:
            log.warning("Could not get git branch")
            log.exception(e)
            branch_name = "UNKNOWN_BRANCH"
    # Try to get current commit
    try:
        commit_sha = repo.head.commit.hexsha
        summary = repo.head.commit.summary
    except ValueError as e:
        if len(list(repo.iter_commits("--all"))) == 0:
            log.warning("No commits in this git repo")
        else:
            log.warning("Could not get commit info")
            log.exception(e)
        commit_sha = "UNKNOWN_SHA"
        summary = "UNKNOWN_SUMMARY"
    log.info(
        "Git repo found [branch {}, Commit {}({})]".format(
            branch_name, commit_sha, summary
        )
    )
    # Check if repo is dirty and log the diff
    dirty = repo.is_dirty()
    if dirty:
        dirty_diff = repo.git.diff()
        log.info("Repo is dirty")
        log.info(f"Dirty repo diff:\n===\n{dirty_diff}\n===")
    return repo
