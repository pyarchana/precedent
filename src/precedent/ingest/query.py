"""The GitHub GraphQL document used to walk a repository's pull requests.

Page sizes are variables rather than literals because GitHub enforces a ~10s
per-query timeout, and deeply nested connections on a repo the size of pandas
will blow through it. The driver backs the sizes down on timeout rather than
giving up on the page.
"""

PR_PAGE_QUERY = """
query PRPage(
  $owner: String!
  $name: String!
  $cursor: String
  $prs: Int!
  $threads: Int!
  $threadComments: Int!
  $reviews: Int!
  $issueComments: Int!
  $files: Int!
) {
  rateLimit {
    limit
    cost
    remaining
    resetAt
  }
  repository(owner: $owner, name: $name) {
    pullRequests(
      first: $prs
      after: $cursor
      orderBy: { field: CREATED_AT, direction: DESC }
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        state
        isDraft
        createdAt
        updatedAt
        mergedAt
        closedAt
        url
        bodyText
        authorAssociation
        author {
          __typename
          login
        }
        labels(first: 20) {
          nodes {
            name
          }
        }
        files(first: $files) {
          pageInfo {
            hasNextPage
          }
          nodes {
            path
            additions
            deletions
            changeType
          }
        }
        reviews(first: $reviews) {
          pageInfo {
            hasNextPage
          }
          nodes {
            id
            state
            submittedAt
            bodyText
            url
            authorAssociation
            author {
              __typename
              login
            }
          }
        }
        reviewThreads(first: $threads) {
          pageInfo {
            hasNextPage
          }
          nodes {
            id
            isResolved
            isOutdated
            path
            line
            originalLine
            diffSide
            comments(first: $threadComments) {
              pageInfo {
                hasNextPage
              }
              nodes {
                id
                databaseId
                bodyText
                createdAt
                url
                path
                originalLine
                diffHunk
                authorAssociation
                author {
                  __typename
                  login
                }
                replyTo {
                  id
                }
              }
            }
          }
        }
        comments(first: $issueComments) {
          pageInfo {
            hasNextPage
          }
          nodes {
            id
            databaseId
            bodyText
            createdAt
            url
            authorAssociation
            author {
              __typename
              login
            }
          }
        }
      }
    }
  }
}
"""

# Fallback for a PR so expensive that even the smallest page times out.
# Fetches nothing but identity, purely so the driver can advance the cursor
# past it and record it for a targeted second pass.
PR_CURSOR_QUERY = """
query PRCursors($owner: String!, $name: String!, $cursor: String, $prs: Int!) {
  rateLimit {
    limit
    cost
    remaining
    resetAt
  }
  repository(owner: $owner, name: $name) {
    pullRequests(
      first: $prs
      after: $cursor
      orderBy: { field: CREATED_AT, direction: DESC }
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
      }
    }
  }
}
"""

# Starting page sizes. The driver halves these on a timeout and restores them
# after a run of clean pages.
DEFAULT_PAGE_SIZES = {
    "prs": 10,
    "threads": 30,
    "threadComments": 30,
    "reviews": 30,
    "issueComments": 30,
    "files": 50,
}

MIN_PAGE_SIZES = {
    "prs": 1,
    "threads": 10,
    "threadComments": 10,
    "reviews": 10,
    "issueComments": 10,
    "files": 20,
}
