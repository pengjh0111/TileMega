/* Does isl represent floor(m / Tm) when Tm is a *parameter*, not a literal?
 * This is the load-bearing question for Part 3: every wait/fanout/count in
 * TileMega's coupling table is a division by a tile size that is symbolic
 * until g is bound. If isl cannot represent a parametric-denominator floor
 * as a genuine Presburger object, the migration needs g bound before isl
 * ever sees the expression, exactly as V-F already found for CuTe's
 * RightInverse.
 */
#include <stdio.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/aff.h>
#include <isl/map.h>
#include <isl/space.h>
#include <isl/local_space.h>
#include <isl/printer.h>
#include <isl/set.h>
#include <barvinok/isl.h>

static void print_map(isl_ctx *ctx, const char *label, __isl_take isl_map *m) {
  isl_printer *p = isl_printer_to_str(ctx);
  p = isl_printer_print_map(p, m);
  char *s = isl_printer_get_str(p);
  fprintf(stderr, "%s: %s\n", label, s);
  free(s);
  isl_printer_free(p);
  isl_map_free(m);
}

int main(void) {
  isl_ctx *ctx = isl_ctx_alloc();

  /* Domain space [S, Tm] -> { [m] } : two parameters, one set dim "m". This
   * is the domain both aff's below are defined over; isl_aff represents
   * (this domain) -> (one scalar expression). */
  isl_space *space = isl_space_set_alloc(ctx, /*nparam=*/2, /*dim=*/1);
  space = isl_space_set_dim_id(space, isl_dim_param, 0, isl_id_alloc(ctx, "S", NULL));
  space = isl_space_set_dim_id(space, isl_dim_param, 1, isl_id_alloc(ctx, "Tm", NULL));

  isl_local_space *ls = isl_local_space_from_space(isl_space_copy(space));
  /* aff1 = m (set dim 0) */
  isl_aff *m = isl_aff_var_on_domain(ls, isl_dim_set, 0);
  /* aff2 = Tm (param dim 1), as an affine expression over the same domain */
  isl_id *tm_id = isl_id_alloc(ctx, "Tm", NULL);
  isl_aff *tm = isl_aff_param_on_domain_space_id(isl_space_copy(space), tm_id);

  isl_aff *quotient = isl_aff_div(m, tm);   /* m / Tm, as a rational aff */
  fprintf(stderr, "quotient (pre-floor) built: %p\n", (void *)quotient);
  isl_aff *floored = isl_aff_floor(isl_aff_copy(quotient));

  isl_printer *p = isl_printer_to_str(ctx);
  p = isl_printer_print_aff(p, floored);
  char *floor_text = isl_printer_get_str(p);
  fprintf(stderr, "floor(m/Tm) as isl_aff = %s\n", floor_text ? floor_text : "(null)");
  free(floor_text);
  isl_printer_free(p);

  isl_map *as_map = isl_map_from_aff(floored);
  print_map(ctx, "floor(m/Tm) as isl_map", isl_map_copy(as_map));

  /* Now count how many m in [0, S) map to a given q -- i.e. is this map
   * usable for card()/is_subset() the way the migration needs? */
  isl_set *dom = isl_set_read_from_str(ctx, "[S,Tm] -> { [m] : 0 <= m < S }");
  isl_map *restricted = isl_map_intersect_domain(as_map, dom);
  isl_pw_qpolynomial *card = isl_map_card(isl_map_copy(restricted));
  p = isl_printer_to_str(ctx);
  p = isl_printer_print_pw_qpolynomial(p, card);
  char *card_text = isl_printer_get_str(p);
  fprintf(stderr, "card(restricted map) = %s\n", card_text ? card_text : "(null)");
  free(card_text);
  isl_printer_free(p);
  isl_pw_qpolynomial_free(card);

  print_map(ctx, "restricted map", restricted);

  isl_aff_free(quotient);
  isl_space_free(space);
  isl_ctx_free(ctx);
  return 0;
}
